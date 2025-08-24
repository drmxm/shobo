#include <rclcpp/rclcpp.hpp>
#include <image_transport/image_transport.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <vision_msgs/msg/detection2_d_array.hpp>
#include <vision_msgs/msg/object_hypothesis_with_pose.hpp>
#include <cv_bridge/cv_bridge.h>

#include <NvInfer.h>
#include <cuda_runtime.h>
#include <opencv2/opencv.hpp>
#include <fstream>
#include <vector>
#include <tuple>
#include <memory>
#include <mutex>
#include <cmath>

using vision_msgs::msg::Detection2DArray;
using vision_msgs::msg::Detection2D;
using vision_msgs::msg::ObjectHypothesisWithPose;

#define CHECK_CUDA(x) do{ cudaError_t e=(x); if(e!=cudaSuccess){throw std::runtime_error(cudaGetErrorString(e));} }while(0)

// Minimal TRT logger
class Logger : public nvinfer1::ILogger {
  void log(Severity s, const char* m) noexcept override {
    if (s <= Severity::kWARNING) fprintf(stderr, "[TRT] %s\n", m);
  }
} gLogger;

// CUDA preproc: BGR8 -> NCHW [0,1], letterbox to (outW,outH)
__global__ void bgr_to_nchw_norm(const unsigned char* __restrict__ bgr,
                                 int inW,int inH,int inStride,
                                 float* __restrict__ out, int outW,int outH,
                                 float sx,float sy, int padX,int padY)
{
  int x = blockIdx.x*blockDim.x + threadIdx.x;
  int y = blockIdx.y*blockDim.y + threadIdx.y;
  if (x>=outW || y>=outH) return;
  int ix = min(inW-1, max(0, int((x - padX)/sx)));
  int iy = min(inH-1, max(0, int((y - padY)/sy)));
  const unsigned char* p = bgr + iy*inStride + ix*3;
  float b = p[0]*(1.f/255.f), g=p[1]*(1.f/255.f), r=p[2]*(1.f/255.f);
  int area = outW*outH;
  out[0*area + y*outW + x] = r;
  out[1*area + y*outW + x] = g;
  out[2*area + y*outW + x] = b;
}

class TrtDetectorNode : public rclcpp::Node {
public:
  TrtDetectorNode() : Node("detector_node")
  {
    auto qos = rclcpp::SensorDataQoS().best_effort().keep_last(5);

    input_topic_     = this->declare_parameter<std::string>("input_topic", "/sensors/rgb/image_raw");
    annotated_topic_ = this->declare_parameter<std::string>("annotated_topic", "/perception/annotated");
    detections_topic_= this->declare_parameter<std::string>("detections_topic", "/perception/detections");
    engine_path_     = this->declare_parameter<std::string>("engine_path", "/work/ros2_ws/yolov8n.engine");
    input_binding_   = this->declare_parameter<std::string>("input_binding", "images");
    output_binding_  = this->declare_parameter<std::string>("output_binding","output0");
    conf_th_         = this->declare_parameter<double>("conf_th", 0.35);
    iou_th_          = this->declare_parameter<double>("iou_th",  0.50);

    it_ = std::make_shared<image_transport::ImageTransport>(shared_from_this());
    sub_ = it_->subscribe(input_topic_, 1, std::bind(&TrtDetectorNode::cb, this, std::placeholders::_1));
    pub_img_ = it_->advertise(annotated_topic_, 1);
    pub_det_ = this->create_publisher<Detection2DArray>(detections_topic_, qos);

    // Load engine
    std::ifstream f(engine_path_, std::ios::binary);
    if(!f) throw std::runtime_error("TRT engine not found at: " + engine_path_);
    f.seekg(0,std::ios::end); size_t sz=f.tellg(); f.seekg(0);
    std::vector<char> buf(sz); f.read(buf.data(), sz);

    runtime_.reset(nvinfer1::createInferRuntime(gLogger));
    engine_.reset(runtime_->deserializeCudaEngine(buf.data(), sz));
    if(!engine_) throw std::runtime_error("deserializeCudaEngine failed");
    ctx_.reset(engine_->createExecutionContext());
    inIndex_  = engine_->getBindingIndex(input_binding_.c_str());
    outIndex_ = engine_->getBindingIndex(output_binding_.c_str());
    inDims_   = engine_->getBindingDimensions(inIndex_);
    outDims_  = engine_->getBindingDimensions(outIndex_);
    if (inDims_.nbDims!=4) throw std::runtime_error("Unexpected input dims");
    netC_ = inDims_.d[1]; netH_ = inDims_.d[2]; netW_ = inDims_.d[3];

    CHECK_CUDA(cudaStreamCreate(&stream_));
    size_t inBytes=netC_*netH_*netW_*sizeof(float);
    size_t outCount=1; for(int i=0;i<outDims_.nbDims;i++) outCount*=outDims_.d[i];
    size_t outBytes= outCount*sizeof(float);
    CHECK_CUDA(cudaMalloc(&dIn_,  inBytes));
    CHECK_CUDA(cudaMalloc(&dOut_, outBytes));
    hOut_.resize(outCount);

    RCLCPP_INFO(get_logger(),"TRT ready: %dx%dx%d → out count %zu", netC_, netH_, netW_, outCount);
  }

  ~TrtDetectorNode(){
    if (dIn_) cudaFree(dIn_);
    if (dOut_) cudaFree(dOut_);
    if (stream_) cudaStreamDestroy(stream_);
  }

private:
  void cb(const sensor_msgs::msg::Image::ConstSharedPtr& msg){
    // BGR frame
    auto cv = cv_bridge::toCvShare(msg, "bgr8");
    const int inW = cv->image.cols, inH = cv->image.rows, inStride = cv->image.step;

    // Register current buffer as mapped (avoid device memcpy). If the pointer changes, re-register.
    size_t bytes = (size_t)inStride * inH;
    if (mapped_host_ != cv->image.data) {
      if (mapped_host_) cudaHostUnregister(mapped_host_);
      CHECK_CUDA(cudaHostRegister(cv->image.data, bytes, cudaHostRegisterMapped));
      mapped_host_ = cv->image.data;
      CHECK_CUDA(cudaHostGetDevicePointer(&mapped_dev_, mapped_host_, 0));
    }

    // Preprocess on GPU
    float sx = std::min((float)netW_/inW, (float)netH_/inH);
    int padX = (int)((netW_ - inW*sx)/2), padY = (int)((netH_ - inH*sx)/2);
    dim3 block(16,16), grid((netW_+15)/16, (netH_+15)/16);
    bgr_to_nchw_norm<<<grid,block,0,stream_>>>(
      (const unsigned char*)mapped_dev_, inW,inH,inStride,
      (float*)dIn_, netW_,netH_, sx,sx, padX,padY);
    CHECK_CUDA(cudaPeekAtLastError());

    // Infer
    void* bindings[2]; bindings[inIndex_] = dIn_; bindings[outIndex_] = dOut_;
    if (!ctx_->enqueueV3(stream_, bindings)){
      RCLCPP_WARN(get_logger(),"TRT enqueue failed"); return;
    }

    // Copy output to host & sync
    size_t outCount = hOut_.size();
    CHECK_CUDA(cudaMemcpyAsync(hOut_.data(), dOut_, outCount*sizeof(float), cudaMemcpyDeviceToHost, stream_));
    CHECK_CUDA(cudaStreamSynchronize(stream_));

    // Decode YOLOv8 head: [1, nAttr=84, nBoxes]
    int nAttr = outDims_.d[1], nBox = outDims_.d[2];
    auto at = [&](int a,int b)->float { return hOut_[a*nBox + b]; };

    std::vector<cv::Rect> rawB; std::vector<int> rawC; std::vector<float> rawS;
    rawB.reserve(200); rawC.reserve(200); rawS.reserve(200);

    for (int b=0;b<nBox;b++){
      float obj = at(4,b);
      int best=-1; float bestp=0.f;
      for(int c=5;c<nAttr;c++){ float p=at(c,b); if(p>bestp){ bestp=p; best=c-5; } }
      float conf = obj*bestp;
      if (conf < conf_th_) continue;

      float cx=at(0,b), cy=at(1,b), w=at(2,b), h=at(3,b);
      // Undo letterbox
      float bx = (cx - w/2.f - padX)/sx;
      float by = (cy - h/2.f - padY)/sx;
      float bw = w/sx, bh = h/sx;
      int x = std::max(0, (int)std::round(bx));
      int y = std::max(0, (int)std::round(by));
      int ww= std::min(inW-x, (int)std::round(bw));
      int hh= std::min(inH-y, (int)std::round(bh));
      rawB.emplace_back(x,y,ww,hh);
      rawC.emplace_back(best);
      rawS.emplace_back(conf);
    }

    // NMS
    std::vector<int> keep; cv::dnn::NMSBoxes(rawB, rawS, (float)conf_th_, (float)iou_th_, keep);

    // Publish detections + annotated image
    Detection2DArray arr; arr.header = msg->header;
    cv::Mat anno = cv->image.clone();
    for (int i : keep){
      const auto& r = rawB[i];
      cv::rectangle(anno, r, {0,255,0}, 2);
      cv::putText(anno, cv::format("%d %.2f", rawC[i], rawS[i]),
                  {r.x, std::max(0,r.y-5)}, cv::FONT_HERSHEY_SIMPLEX, 0.5, {0,255,0}, 1);
      Detection2D d;
      d.bbox.center.x = r.x + r.width/2.0;
      d.bbox.center.y = r.y + r.height/2.0;
      d.bbox.size_x = r.width; d.bbox.size_y = r.height;
      ObjectHypothesisWithPose hyp;
      hyp.hypothesis.class_id = std::to_string(rawC[i]);
      hyp.hypothesis.score = rawS[i];
      d.results.push_back(hyp);
      arr.detections.push_back(d);
    }
    pub_det_->publish(arr);
    pub_img_.publish(cv_bridge::CvImage(msg->header, "bgr8", anno).toImageMsg());
  }

  // Members
  std::shared_ptr<image_transport::ImageTransport> it_;
  image_transport::Subscriber sub_;
  image_transport::Publisher  pub_img_;
  rclcpp::Publisher<Detection2DArray>::SharedPtr pub_det_;

  std::string input_topic_, annotated_topic_, detections_topic_, engine_path_;
  std::string input_binding_, output_binding_;
  double conf_th_{0.35}, iou_th_{0.5};

  // TRT
  std::unique_ptr<nvinfer1::IRuntime> runtime_{};
  std::unique_ptr<nvinfer1::ICudaEngine> engine_{};
  std::unique_ptr<nvinfer1::IExecutionContext> ctx_{};
  int inIndex_{-1}, outIndex_{-1};
  nvinfer1::Dims inDims_{}, outDims_{};
  int netC_{3}, netH_{640}, netW_{640};
  void* dIn_{nullptr};
  void* dOut_{nullptr};
  cudaStream_t stream_{};
  std::vector<float> hOut_;

  // mapped host buffer alias (zero-copy into GPU address space)
  unsigned char* mapped_host_{nullptr};
  unsigned char* mapped_dev_{nullptr};
};

int main(int argc,char** argv){
  rclcpp::init(argc,argv);
  rclcpp::spin(std::make_shared<TrtDetectorNode>());
  rclcpp::shutdown();
  return 0;
}
