// #include <rclcpp/rclcpp.hpp>
// #include <image_transport/image_transport.hpp>
// #include <sensor_msgs/msg/image.hpp>

// #include <vision_msgs/msg/detection2_d.hpp>
// #include <vision_msgs/msg/detection2_d_array.hpp>
// #include <vision_msgs/msg/object_hypothesis_with_pose.hpp>

// #include <vision_msgs/msg/pose2_d.hpp>

// #include <cv_bridge/cv_bridge.h>

// #include <NvInfer.h>
// #include <cuda_runtime.h>
// #include <opencv2/opencv.hpp>
// #include <fstream>
// #include <vector>
// #include <memory>
// #include <algorithm>
// #include <string>

// #include "shobo_detectors/kernels.hpp"

// using vision_msgs::msg::Detection2DArray;
// using vision_msgs::msg::Detection2D;
// using vision_msgs::msg::ObjectHypothesisWithPose;

// using vision_msgs::msg::Pose2D;

// #define CHECK_CUDA(x) do{ cudaError_t e=(x); if(e!=cudaSuccess){throw std::runtime_error(cudaGetErrorString(e));} }while(0)

// class Logger : public nvinfer1::ILogger {
//   void log(Severity s, const char* m) noexcept override {
//     if (s <= Severity::kWARNING) fprintf(stderr, "[TRT] %s\n", m);
//   }
// } gLogger;

// class TrtDetectorNode : public rclcpp::Node {
// public:
//   TrtDetectorNode() : Node("detector_node")
//   {
//     auto qos = rclcpp::SensorDataQoS().best_effort().keep_last(5);

//     input_topic_     = this->declare_parameter<std::string>("input_topic", "/sensors/rgb/image_raw");
//     annotated_topic_ = this->declare_parameter<std::string>("annotated_topic", "/perception/annotated");
//     detections_topic_= this->declare_parameter<std::string>("detections_topic", "/perception/detections");
//     engine_path_     = this->declare_parameter<std::string>("engine_path", "/work/ros2_ws/yolov8n.engine");
//     input_binding_   = this->declare_parameter<std::string>("input_binding", "images");
//     output_binding_  = this->declare_parameter<std::string>("output_binding","output0");
//     conf_th_         = this->declare_parameter<double>("conf_th", 0.35);
//     iou_th_          = this->declare_parameter<double>("iou_th",  0.50);

//     it_ = std::make_shared<image_transport::ImageTransport>(shared_from_this());
//     sub_ = it_->subscribe(input_topic_, 1, std::bind(&TrtDetectorNode::cb, this, std::placeholders::_1));
//     pub_img_ = it_->advertise(annotated_topic_, 1);
//     pub_det_ = this->create_publisher<Detection2DArray>(detections_topic_, qos);

//     // Load TRT engine
//     std::ifstream f(engine_path_, std::ios::binary);
//     if(!f) throw std::runtime_error("TRT engine not found at: " + engine_path_);
//     f.seekg(0,std::ios::end); size_t sz=f.tellg(); f.seekg(0);
//     std::vector<char> buf(sz); f.read(buf.data(), sz);

//     runtime_.reset(nvinfer1::createInferRuntime(gLogger));
//     engine_.reset(runtime_->deserializeCudaEngine(buf.data(), sz));
//     if(!engine_) throw std::runtime_error("deserializeCudaEngine failed");
//     ctx_.reset(engine_->createExecutionContext());

//     // ---- IO-BY-NAME API (TRT 8.6+) ----
//     const int nIO = engine_->getNbIOTensors();
//     for (int i=0;i<nIO;i++){
//       const char* nm = engine_->getIOTensorName(i);
//       auto mode = engine_->getTensorIOMode(nm);
//       if (mode == nvinfer1::TensorIOMode::kINPUT)  inputs_.push_back(nm);
//       if (mode == nvinfer1::TensorIOMode::kOUTPUT) outputs_.push_back(nm);
//     }
//     if (std::find(inputs_.begin(), inputs_.end(), input_binding_) == inputs_.end())
//       input_binding_  = inputs_.empty() ? "" : inputs_.front();
//     if (std::find(outputs_.begin(), outputs_.end(), output_binding_) == outputs_.end())
//       output_binding_ = outputs_.empty() ? "" : outputs_.front();
//     if (input_binding_.empty() || output_binding_.empty())
//       throw std::runtime_error("Could not resolve TRT I/O tensor names");

//     // Input dims expected as [N,C,H,W]
//     inDims_ = engine_->getTensorShape(input_binding_.c_str());
//     if (inDims_.nbDims != 4) throw std::runtime_error("Unexpected input dims (expect NCHW)");
//     netC_ = inDims_.d[1]; netH_ = inDims_.d[2]; netW_ = inDims_.d[3];

//     outDims_ = engine_->getTensorShape(output_binding_.c_str());
//     size_t outCount = 1; for (int i=0;i<outDims_.nbDims;i++) outCount *= outDims_.d[i];

//     CHECK_CUDA(cudaStreamCreate(&stream_));
//     size_t inBytes  = (size_t)netC_*netH_*netW_*sizeof(float);
//     size_t outBytes = outCount*sizeof(float);
//     CHECK_CUDA(cudaMalloc(&dIn_,  inBytes));
//     CHECK_CUDA(cudaMalloc(&dOut_, outBytes));
//     hOut_.resize(outCount);

//     RCLCPP_INFO(get_logger(),
//       "TRT ready: input=%s (%dx%dx%d), output=%s, outCount=%zu",
//       input_binding_.c_str(), netC_, netH_, netW_, output_binding_.c_str(), outCount);
//   }

//   ~TrtDetectorNode(){
//     if (mapped_host_) cudaHostUnregister(mapped_host_);
//     if (dIn_)  cudaFree(dIn_);
//     if (dOut_) cudaFree(dOut_);
//     if (stream_) cudaStreamDestroy(stream_);
//   }

// private:
//   void cb(const sensor_msgs::msg::Image::ConstSharedPtr& msg){
//     auto cv = cv_bridge::toCvShare(msg, "bgr8");
//     const int inW = cv->image.cols, inH = cv->image.rows, inStride = (int)cv->image.step;

//     // Pinned zero-copy map for current frame
//     size_t bytes = (size_t)inStride * inH;
//     if (mapped_host_ != cv->image.data) {
//       if (mapped_host_) cudaHostUnregister(mapped_host_);
//       CHECK_CUDA(cudaHostRegister(cv->image.data, bytes, cudaHostRegisterMapped));
//       mapped_host_ = cv->image.data;
//       CHECK_CUDA(cudaHostGetDevicePointer(&mapped_dev_, mapped_host_, 0));
//     }

//     // Letterbox scale to net input
//     float sx = std::min((float)netW_/inW, (float)netH_/inH);
//     int padX = (int)((netW_ - inW*sx)/2), padY = (int)((netH_ - inH*sx)/2);

//     // CUDA preprocess
//     launch_bgr_to_nchw_norm((const unsigned char*)mapped_dev_, inW,inH,inStride,
//                             (float*)dIn_, netW_,netH_, sx,sx, padX,padY, stream_);

//     // Bind I/O by name and run
//     ctx_->setTensorAddress(input_binding_.c_str(),  dIn_);
//     ctx_->setTensorAddress(output_binding_.c_str(), dOut_);
//     if (!ctx_->enqueueV3(stream_)) {
//       RCLCPP_WARN(get_logger(),"TRT enqueueV3 failed");
//       return;
//     }

//     // Copy output back
//     CHECK_CUDA(cudaMemcpyAsync(hOut_.data(), dOut_, hOut_.size()*sizeof(float),
//                                cudaMemcpyDeviceToHost, stream_));
//     CHECK_CUDA(cudaStreamSynchronize(stream_));

//     // YOLOv8 head: [1, nAttr, nBox]
//     const int nAttr = outDims_.d[1];
//     const int nBox  = outDims_.d[2];
//     auto at = [&](int a,int b)->float { return hOut_[a*nBox + b]; };

//     std::vector<cv::Rect> rawB; std::vector<int> rawC; std::vector<float> rawS;
//     rawB.reserve(256); rawC.reserve(256); rawS.reserve(256);

//     for (int b=0;b<nBox;b++){
//       float obj = at(4,b);
//       int best=-1; float bestp=0.f;
//       for(int c=5;c<nAttr;c++){ float p=at(c,b); if(p>bestp){ bestp=p; best=c-5; } }
//       float conf = obj*bestp;
//       if (conf < conf_th_) continue;

//       float cx=at(0,b), cy=at(1,b), w=at(2,b), h=at(3,b);
//       // Undo letterbox
//       float bx = (cx - w/2.f - padX)/sx;
//       float by = (cy - h/2.f - padY)/sx;
//       float bw = w/sx, bh = h/sx;
//       int x = std::max(0, (int)std::round(bx));
//       int y = std::max(0, (int)std::round(by));
//       int ww= std::min(inW-x, (int)std::round(bw));
//       int hh= std::min(inH-y, (int)std::round(bh));
//       rawB.emplace_back(x,y,ww,hh);
//       rawC.emplace_back(best);
//       rawS.emplace_back(conf);
//     }

//     std::vector<int> keep; cv::dnn::NMSBoxes(rawB, rawS, (float)conf_th_, (float)iou_th_, keep);

//     // Publish
//     Detection2DArray arr; arr.header = msg->header;
//     cv::Mat anno = cv->image.clone();
//     for (int i : keep){
//       const auto& r = rawB[i];
//       cv::rectangle(anno, r, {0,255,0}, 2);
//       cv::putText(anno, cv::format("%d %.2f", rawC[i], rawS[i]),
//                   {r.x, std::max(0,r.y-5)}, cv::FONT_HERSHEY_SIMPLEX, 0.5, {0,255,0}, 1);

//       Detection2D d;
//       Pose2D c;
//       c.position.x = r.x + r.width * 0.5;
//       c.position.y = r.y + r.height * 0.5;
//       c.theta = 0.0;
//       d.bbox.center = c;
//       d.bbox.size_x = r.width; d.bbox.size_y = r.height;

//       ObjectHypothesisWithPose hyp;
//       hyp.hypothesis.class_id = std::to_string(rawC[i]);
//       hyp.hypothesis.score = rawS[i];
//       d.results.push_back(hyp);
//       arr.detections.push_back(d);
//     }
//     pub_det_->publish(arr);
//     pub_img_.publish(cv_bridge::CvImage(msg->header, "bgr8", anno).toImageMsg());
//   }

//   // ROS
//   std::shared_ptr<image_transport::ImageTransport> it_;
//   image_transport::Subscriber sub_;
//   image_transport::Publisher  pub_img_;
//   rclcpp::Publisher<Detection2DArray>::SharedPtr pub_det_;

//   // params
//   std::string input_topic_, annotated_topic_, detections_topic_, engine_path_;
//   std::string input_binding_, output_binding_;
//   double conf_th_{0.35}, iou_th_{0.5};

//   // TRT
//   std::unique_ptr<nvinfer1::IRuntime> runtime_{};
//   std::unique_ptr<nvinfer1::ICudaEngine> engine_{};
//   std::unique_ptr<nvinfer1::IExecutionContext> ctx_{};
//   nvinfer1::Dims inDims_{}, outDims_{};
//   int netC_{3}, netH_{640}, netW_{640};
//   std::vector<std::string> inputs_, outputs_;
//   void* dIn_{nullptr};
//   void* dOut_{nullptr};
//   cudaStream_t stream_{};
//   std::vector<float> hOut_;

//   // pinned zero-copy alias
//   unsigned char* mapped_host_{nullptr};
//   unsigned char* mapped_dev_{nullptr};
// };

// int main(int argc,char** argv){
//   rclcpp::init(argc,argv);
//   rclcpp::spin(std::make_shared<TrtDetectorNode>());
//   rclcpp::shutdown();
//   return 0;
// }
#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/image.hpp>

// image_transport headers are available via package deps,
// here we just use plain rclcpp subscription for simplicity.
#if __has_include(<vision_msgs/msg/detection2_d.hpp>)
  #include <vision_msgs/msg/detection2_d.hpp>
  namespace vmsg = vision_msgs::msg;
#elif __has_include(<vision_msgs/msg/detection2d.hpp>)
  #include <vision_msgs/msg/detection2d.hpp>
  namespace vmsg = vision_msgs::msg;
#else
  #pragma message("vision_msgs Detection2D header not found; continuing without it.")
#endif

// Include TensorRT only if CMake found it (and defined SHB_USE_TRT)
#ifdef SHB_USE_TRT
  #include <NvInfer.h>
  #pragma message("Building with TensorRT enabled (SHB_USE_TRT).")
#else
  #pragma message("Building WITHOUT TensorRT (SHB_USE_TRT not defined).")
#endif

#include "shobo_detectors/kernels.hpp"

class DetectorNode : public rclcpp::Node {
public:
  DetectorNode() : rclcpp::Node("detector_node") {
    sub_ = this->create_subscription<sensor_msgs::msg::Image>(
      "image_in", rclcpp::SensorDataQoS(),
      std::bind(&DetectorNode::onImage, this, std::placeholders::_1));
    pub_ = this->create_publisher<sensor_msgs::msg::Image>("image_out", 10);
    RCLCPP_INFO(get_logger(), "detector_node up.");
  }

private:
  void onImage(const sensor_msgs::msg::Image::SharedPtr msg) {
    // Minimal pass-through to prove build & runtime wiring
    (void)msg;
    // Normally you’d preprocess (maybe via CUDA kernels) and run TRT here.
  }

  rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr sub_;
  rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr pub_;
};

int main(int argc, char** argv) {
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<DetectorNode>());
  rclcpp::shutdown();
  return 0;
}
