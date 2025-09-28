#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <image_transport/image_transport.hpp>
#include <cv_bridge/cv_bridge.h>
#include <opencv2/opencv.hpp>

#if __has_include(<vision_msgs/msg/detection2_d.hpp>)
  #include <vision_msgs/msg/detection2_d.hpp>
  #include <vision_msgs/msg/detection2_d_array.hpp>
  #include <vision_msgs/msg/object_hypothesis_with_pose.hpp>
  namespace vision = vision_msgs::msg;
#elif __has_include(<vision_msgs/msg/detection2d.hpp>)
  #include <vision_msgs/msg/detection2d.hpp>
  #include <vision_msgs/msg/detection2d_array.hpp>
  #include <vision_msgs/msg/object_hypothesis_with_pose.hpp>
  namespace vision = vision_msgs::msg;
#else
  #error "vision_msgs Detection2D message not found. Install ros-humble-vision-msgs."
#endif

#include "shobo_detectors/kernels.hpp"  // keep include; we hard-guard usage

#if SHB_USE_TRT

#include <NvInfer.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <memory>
#include <numeric>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

class Logger : public nvinfer1::ILogger {
public:
  void log(Severity s, const char* msg) noexcept override {
    auto lg = rclcpp::get_logger("trt_detector");
    switch (s) {
      case Severity::kINTERNAL_ERROR:
      case Severity::kERROR:   RCLCPP_ERROR(lg, "[TRT] %s", msg); break;
      case Severity::kWARNING: RCLCPP_WARN (lg, "[TRT] %s", msg); break;
      case Severity::kINFO:    RCLCPP_INFO (lg, "[TRT] %s", msg); break;
      default:                 RCLCPP_DEBUG(lg, "[TRT] %s", msg); break;
    }
  }
};
Logger gLogger;

#define CHECK_CUDA(cmd) do {                                         \
  cudaError_t e__ = (cmd);                                           \
  if (e__ != cudaSuccess) {                                          \
    throw std::runtime_error(std::string("CUDA error: ") +           \
                             cudaGetErrorString(e__));               \
  }                                                                  \
} while (0)

size_t element_size(nvinfer1::DataType t) {
  switch (t) {
    case nvinfer1::DataType::kFLOAT: return sizeof(float);
    case nvinfer1::DataType::kHALF:  return sizeof(uint16_t);
    case nvinfer1::DataType::kINT8:
    case nvinfer1::DataType::kBOOL:  return sizeof(uint8_t);
    case nvinfer1::DataType::kINT32: return sizeof(int32_t);
    case nvinfer1::DataType::kINT64: return sizeof(int64_t);
    case nvinfer1::DataType::kUINT8: return sizeof(uint8_t);
    default: throw std::runtime_error("Unsupported TRT dtype");
  }
}

std::string dims_to_string(const nvinfer1::Dims& d) {
  std::ostringstream oss;
  oss << "[";
  for (int i = 0; i < d.nbDims; ++i) { if (i) oss << ","; oss << d.d[i]; }
  oss << "]";
  return oss.str();
}

const char* dtype_to_string(nvinfer1::DataType t) {
  switch (t) {
    case nvinfer1::DataType::kFLOAT: return "float32";
    case nvinfer1::DataType::kHALF:  return "float16";
    case nvinfer1::DataType::kINT8:  return "int8";
    case nvinfer1::DataType::kINT32: return "int32";
    case nvinfer1::DataType::kBOOL:  return "bool";
    case nvinfer1::DataType::kUINT8: return "uint8";
    case nvinfer1::DataType::kINT64: return "int64";
    default: return "unknown";
  }
}

float iou(const cv::Rect& a, const cv::Rect& b) {
  const int x1 = std::max(a.x, b.x);
  const int y1 = std::max(a.y, b.y);
  const int x2 = std::min(a.x + a.width,  b.x + b.width);
  const int y2 = std::min(a.y + a.height, b.y + b.height);
  const int iw = std::max(0, x2 - x1);
  const int ih = std::max(0, y2 - y1);
  const int inter = iw * ih;
  if (inter <= 0) return 0.f;
  const int uni = a.area() + b.area() - inter;
  return uni > 0 ? float(inter) / float(uni) : 0.f;
}

void nms(const std::vector<cv::Rect>& boxes,
         const std::vector<float>& scores,
         float thr,
         std::vector<int>& keep) {
  const size_t n = boxes.size();
  keep.clear();
  if (!n) return;
  std::vector<int> order(n); std::iota(order.begin(), order.end(), 0);
  std::sort(order.begin(), order.end(),
            [&](int i, int j){ return scores[i] > scores[j]; });
  std::vector<char> sup(n, 0);
  for (size_t i = 0; i < n; ++i) {
    int idx = order[i];
    if (sup[idx]) continue;
    keep.push_back(idx);
    for (size_t j = i + 1; j < n; ++j) {
      int k = order[j];
      if (!sup[k] && iou(boxes[idx], boxes[k]) >= thr) sup[k] = 1;
    }
  }
}

} // namespace

class TrtDetectorNode : public rclcpp::Node {
public:
  TrtDetectorNode() : rclcpp::Node("trt_detector") {
    rclcpp::SensorDataQoS qos; qos.keep_last(5); qos.best_effort();

    input_topic_      = declare_parameter<std::string>("input_topic",      "/rgb_cam/image_raw");
    annotated_topic_  = declare_parameter<std::string>("annotated_topic",  "/perception/annotated");
    detections_topic_ = declare_parameter<std::string>("detections_topic", "/perception/detections");
    publish_annotated_= declare_parameter<bool>("publish_annotated", true);

    engine_path_      = declare_parameter<std::string>("engine_path",      "/work/ros2_ws/yolov8n.engine");
    input_binding_    = declare_parameter<std::string>("input_binding",    "images");
    output_binding_   = declare_parameter<std::string>("output_binding",   "output0");

    conf_th_          = declare_parameter<double>("conf_th", 0.35);
    iou_th_           = declare_parameter<double>("iou_th",  0.50);
    max_detections_   = declare_parameter<int>("max_detections", 300);
    use_cuda_preproc_ = declare_parameter<bool>("use_cuda_preproc", false); // default safe
    class_labels_     = declare_parameter<std::vector<std::string>>("class_labels", {});

    load_engine_and_allocate();

    sub_ = create_subscription<sensor_msgs::msg::Image>(
      input_topic_, qos, std::bind(&TrtDetectorNode::on_image, this, std::placeholders::_1));

    if (publish_annotated_) {
      pub_annotated_ = image_transport::create_publisher(this, annotated_topic_);
    }
    pub_detections_ = create_publisher<vision::Detection2DArray>(detections_topic_, qos);

    RCLCPP_INFO(get_logger(), "TensorRT detector ready: input=%s output=%s dims=[%d,%d,%d]",
                input_binding_.c_str(), output_binding_.c_str(), netC_, netH_, netW_);
  }

  ~TrtDetectorNode() override {
    try {
      if (stream_) cudaStreamDestroy(stream_);
      if (dIn_)    cudaFree(dIn_);
      for (auto& b : outputs_) { if (b.device) cudaFree(b.device); }
    } catch (...) {}
  }

private:
  struct OutBuf {
    std::string name;
    nvinfer1::Dims dims{};
    nvinfer1::DataType dtype{nvinfer1::DataType::kFLOAT};
    void* device{nullptr};
    size_t bytes{0};
  };

  void load_engine_and_allocate() {
    // Read engine
    std::ifstream f(engine_path_, std::ios::binary);
    if (!f) throw std::runtime_error("Engine not found at " + engine_path_);
    f.seekg(0, std::ios::end); size_t sz = static_cast<size_t>(f.tellg());
    f.seekg(0, std::ios::beg); std::vector<char> buff(sz); f.read(buff.data(), sz);

    runtime_.reset(nvinfer1::createInferRuntime(gLogger));
    if (!runtime_) throw std::runtime_error("Failed to create TRT runtime");
    engine_.reset(runtime_->deserializeCudaEngine(buff.data(), sz));
    if (!engine_) throw std::runtime_error("Failed to deserialize engine");
    context_.reset(engine_->createExecutionContext());
    if (!context_) throw std::runtime_error("Failed to create execution context");

    // IO names
    std::vector<std::string> in_names, out_names;
    const int nb = engine_->getNbIOTensors();
    for (int i = 0; i < nb; ++i) {
      const char* n = engine_->getIOTensorName(i);
      auto mode = engine_->getTensorIOMode(n);
      if (mode == nvinfer1::TensorIOMode::kINPUT)  in_names.emplace_back(n);
      else                                         out_names.emplace_back(n);
    }

    if (std::find(in_names.begin(), in_names.end(), input_binding_) == in_names.end()) {
      if (in_names.empty()) throw std::runtime_error("Engine has no inputs");
      RCLCPP_WARN(get_logger(), "Input binding '%s' not found. Using '%s'.",
                  input_binding_.c_str(), in_names.front().c_str());
      input_binding_ = in_names.front();
    }
    if (std::find(out_names.begin(), out_names.end(), output_binding_) == out_names.end()) {
      if (out_names.empty()) throw std::runtime_error("Engine has no outputs");
      RCLCPP_WARN(get_logger(), "Output binding '%s' not found. Using '%s'.",
                  output_binding_.c_str(), out_names.front().c_str());
      output_binding_ = out_names.front();
    }

    // Input dims from engine, then set real shape on context
    nvinfer1::Dims in_engine = engine_->getTensorShape(input_binding_.c_str());
    if (in_engine.nbDims != 4) throw std::runtime_error("Expected NCHW input");

    // Resolve dynamic batch if any -> set [1, C, H, W]
    netN_ = (in_engine.d[0] > 0) ? in_engine.d[0] : 1;
    netC_ = (in_engine.d[1] > 0) ? in_engine.d[1] : 3;
    netH_ = (in_engine.d[2] > 0) ? in_engine.d[2] : 640;
    netW_ = (in_engine.d[3] > 0) ? in_engine.d[3] : 640;

    input_dims_ = nvinfer1::Dims4{netN_, netC_, netH_, netW_};

    input_type_ = engine_->getTensorDataType(input_binding_.c_str());
    if (input_type_ != nvinfer1::DataType::kFLOAT &&
        input_type_ != nvinfer1::DataType::kHALF) {
      throw std::runtime_error("Unsupported input dtype");
    }

    CHECK_CUDA(cudaStreamCreate(&stream_));
    if (!context_->setOptimizationProfileAsync(0, stream_))
      throw std::runtime_error("Failed to set optimization profile");
    if (!context_->setInputShape(input_binding_.c_str(), input_dims_))
      throw std::runtime_error("Failed to set input shape");
    CHECK_CUDA(cudaStreamSynchronize(stream_));

    // Now use the **resolved shapes** from the context.
    nvinfer1::Dims in_ctx = context_->getTensorShape(input_binding_.c_str());
    netN_ = in_ctx.d[0]; netC_ = in_ctx.d[1]; netH_ = in_ctx.d[2]; netW_ = in_ctx.d[3];

    input_count_ = static_cast<size_t>(netC_) * netH_ * netW_;
    inBytes_     = input_count_ * element_size(input_type_);

    CHECK_CUDA(cudaMalloc(&dIn_, inBytes_));

    RCLCPP_INFO(get_logger(), "[%s] TensorRT input tensor %s dims=%s dtype=%s (%zu bytes)",
                get_name(), input_binding_.c_str(), dims_to_string(in_ctx).c_str(),
                dtype_to_string(input_type_), inBytes_);

    // Allocate outputs using context-resolved dims
    outputs_.clear();
    dOut_ = nullptr; outBytes_ = 0; outCount_ = 0; output_type_ = nvinfer1::DataType::kFLOAT;

    for (const auto& n : out_names) {
      OutBuf ob;
      ob.name  = n;
      ob.dims  = context_->getTensorShape(n.c_str());
      ob.dtype = engine_->getTensorDataType(n.c_str());

      size_t count = 1;
      for (int i = 0; i < ob.dims.nbDims; ++i) {
        if (ob.dims.d[i] <= 0) throw std::runtime_error("Dynamic output needs shape bound: " + n);
        count *= static_cast<size_t>(ob.dims.d[i]);
      }
      ob.bytes = count * element_size(ob.dtype);
      CHECK_CUDA(cudaMalloc(&ob.device, ob.bytes));
      outputs_.push_back(ob);

      if (n == output_binding_) {
        outDims_     = ob.dims;
        outCount_    = static_cast<int>(count);
        outBytes_    = ob.bytes;
        output_type_ = ob.dtype;
        dOut_        = ob.device;
      }

      RCLCPP_INFO(get_logger(), "[%s] TensorRT output tensor %s dims=%s dtype=%s (%zu bytes)",
                  get_name(), n.c_str(), dims_to_string(ob.dims).c_str(),
                  dtype_to_string(ob.dtype), ob.bytes);
    }
    if (!dOut_) throw std::runtime_error("No output buffer for '" + output_binding_ + "'");

    // Host buffers
    hostInputF_.resize(input_count_);
    hostInputH_.resize(input_count_);
    hostOutputF_.resize(outCount_);
    hostOutputRaw_.resize(outBytes_);

    // Safety: disable GPU preproc unless engine input is float32 and user requested it
    if (input_type_ != nvinfer1::DataType::kFLOAT && use_cuda_preproc_) {
      RCLCPP_WARN(get_logger(), "Disabling CUDA preproc: engine input dtype is %s",
                  dtype_to_string(input_type_));
      use_cuda_preproc_ = false;
    }
  }

  void on_image(const sensor_msgs::msg::Image::ConstSharedPtr& msg) {
    cv_bridge::CvImageConstPtr cv_ptr;
    try {
      cv_ptr = cv_bridge::toCvShare(msg, "bgr8");
    } catch (const cv_bridge::Exception& e) {
      RCLCPP_WARN_THROTTLE(get_logger(), *this->get_clock(), 2000, "cv_bridge: %s", e.what());
      return;
    }
    const cv::Mat& bgr = cv_ptr->image;
    if (bgr.empty()) return;

    // Letterbox → NCHW float
    const float scale = std::min(float(netW_) / bgr.cols, float(netH_) / bgr.rows);
    const int   newW  = static_cast<int>(std::round(bgr.cols * scale));
    const int   newH  = static_cast<int>(std::round(bgr.rows * scale));
    const int   padX  = (netW_ - newW) / 2;
    const int   padY  = (netH_ - newH) / 2;

    if (!preprocess(bgr, scale, padX, padY)) return;

    // Clear any sticky CUDA error from prior ops BEFORE enqueue.
    (void)cudaGetLastError();

    if (!context_->setInputShape(input_binding_.c_str(), input_dims_)) {
      RCLCPP_ERROR(get_logger(), "Failed to set input shape before enqueue");
      return;
    }

    context_->setTensorAddress(input_binding_.c_str(), dIn_);
    for (auto& ob : outputs_) {
      context_->setTensorAddress(ob.name.c_str(), ob.device);
    }

    if (!context_->enqueueV3(stream_)) {
      cudaError_t e = cudaGetLastError(); // capture CUDA state
      (void)cudaStreamSynchronize(stream_);
      RCLCPP_WARN_THROTTLE(get_logger(), *this->get_clock(), 2000,
                           "TensorRT enqueueV3 failed (CUDA: %s)", cudaGetErrorString(e));
      return;
    }

    // D2H
    if (output_type_ == nvinfer1::DataType::kFLOAT) {
      CHECK_CUDA(cudaMemcpyAsync(hostOutputF_.data(), dOut_, outBytes_, cudaMemcpyDeviceToHost, stream_));
    } else if (output_type_ == nvinfer1::DataType::kHALF) {
      CHECK_CUDA(cudaMemcpyAsync(hostOutputRaw_.data(), dOut_, outBytes_, cudaMemcpyDeviceToHost, stream_));
    } else {
      RCLCPP_ERROR(get_logger(), "Unsupported output dtype");
      return;
    }
    CHECK_CUDA(cudaStreamSynchronize(stream_));
    if (output_type_ == nvinfer1::DataType::kHALF) {
      const uint16_t* src = reinterpret_cast<const uint16_t*>(hostOutputRaw_.data());
      for (int i = 0; i < outCount_; ++i) {
        __half h; std::memcpy(&h, &src[i], sizeof(uint16_t));
        hostOutputF_[i] = __half2float(h);
      }
    }

    publish_detections(msg, bgr, scale, padX, padY);
  }

  bool preprocess(const cv::Mat& bgr, float scale, int padX, int padY) {
    // CPU path is safe and fast for 640x640 on Orin; keep CUDA path hard-guarded.
    if (use_cuda_preproc_) {
      try {
        // The custom kernel MUST write float32 NCHW with letterbox. If it isn’t 100% correct,
        // disable it via YAML to avoid illegal memory access.
        shobo::bgr_to_nchw_norm(
          bgr.data, bgr.cols, bgr.rows, static_cast<int>(bgr.step),
          reinterpret_cast<float*>(dIn_), netW_, netH_,
          scale, scale, padX, padY, stream_);
        CHECK_CUDA(cudaPeekAtLastError());
        CHECK_CUDA(cudaStreamSynchronize(stream_));
        return true;
      } catch (const std::exception& e) {
        RCLCPP_WARN(get_logger(), "CUDA preprocess failed: %s. Falling back to CPU.", e.what());
        use_cuda_preproc_ = false; // permanently disable this run
      }
    }

    // CPU letterbox → RGB → float32 NCHW in [0,1]
    cv::Mat resized; cv::resize(bgr, resized, cv::Size(), scale, scale, cv::INTER_LINEAR);
    cv::Mat canvas(netH_, netW_, CV_8UC3, cv::Scalar(114,114,114));
    resized.copyTo(canvas(cv::Rect(padX, padY, resized.cols, resized.rows)));

    cv::Mat rgb; cv::cvtColor(canvas, rgb, cv::COLOR_BGR2RGB);
    cv::Mat f;   rgb.convertTo(f, CV_32FC3, 1.f/255.f);

    const int channelSize = netH_ * netW_;
    std::vector<cv::Mat> ch(3); cv::split(f, ch);

    for (int c = 0; c < 3; ++c) {
      std::memcpy(hostInputF_.data() + c * channelSize, ch[c].data, channelSize * sizeof(float));
    }

    if (input_type_ == nvinfer1::DataType::kFLOAT) {
      CHECK_CUDA(cudaMemcpyAsync(dIn_, hostInputF_.data(), inBytes_, cudaMemcpyHostToDevice, stream_));
    } else { // HALF
      for (size_t i = 0; i < input_count_; ++i) {
        hostInputH_[i] = __float2half(hostInputF_[i]);
      }
      CHECK_CUDA(cudaMemcpyAsync(dIn_, hostInputH_.data(), inBytes_, cudaMemcpyHostToDevice, stream_));
    }
    CHECK_CUDA(cudaStreamSynchronize(stream_));
    return true;
  }

  void publish_detections(const sensor_msgs::msg::Image::ConstSharedPtr& msg,
                          const cv::Mat& frame, float scale, int padX, int padY) {
    const int nAttr  = outDims_.d[1];  // 84 for YOLOv8 COCO
    const int nBoxes = outDims_.d[2];  // 8400 for 640x640

    auto at = [&](int a, int b)->float { return hostOutputF_[a * nBoxes + b]; };

    std::vector<cv::Rect> boxes; boxes.reserve(nBoxes);
    std::vector<int>      cls;   cls.reserve(nBoxes);
    std::vector<float>    scr;   scr.reserve(nBoxes);

    for (int b = 0; b < nBoxes; ++b) {
      const float obj = at(4, b);
      int bestC = -1; float bestP = 0.f;
      for (int c = 5; c < nAttr; ++c) {
        const float p = at(c, b);
        if (p > bestP) { bestP = p; bestC = c - 5; }
      }
      const float conf = obj * bestP;
      if (conf < static_cast<float>(conf_th_)) continue;

      const float cx = at(0, b), cy = at(1, b);
      const float w  = at(2, b), h  = at(3, b);

      const float left = (cx - 0.5f * w - padX) / scale;
      const float top  = (cy - 0.5f * h - padY) / scale;
      const float ww   =  w / scale, hh = h / scale;

      int x = std::max(0, static_cast<int>(std::round(left)));
      int y = std::max(0, static_cast<int>(std::round(top)));
      int X = std::min(frame.cols, static_cast<int>(std::round(left + ww)));
      int Y = std::min(frame.rows, static_cast<int>(std::round(top  + hh)));
      int W = std::max(0, X - x);
      int H = std::max(0, Y - y);
      if (W <= 0 || H <= 0) continue;

      boxes.emplace_back(x, y, W, H);
      cls.emplace_back(bestC);
      scr.emplace_back(conf);
    }

    std::vector<int> keep; nms(boxes, scr, static_cast<float>(iou_th_), keep);
    if (max_detections_ > 0 && static_cast<int>(keep.size()) > max_detections_)
      keep.resize(static_cast<size_t>(max_detections_));

    vision::Detection2DArray out; out.header = msg->header;
    cv::Mat annotated = frame.clone();

    for (int idx : keep) {
      const auto& r = boxes[idx]; const float s = scr[idx]; const int c = cls[idx];

      if (publish_annotated_) {
        cv::rectangle(annotated, r, cv::Scalar(0,255,0), 2);
        std::ostringstream lab; lab.setf(std::ios::fixed); lab.precision(2);
        lab << class_name(c) << " " << s;
        cv::putText(annotated, lab.str(), cv::Point(r.x, std::max(0, r.y - 5)),
                    cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0,255,0), 1);
      }

      vision::Detection2D d;
      d.bbox.center.position.x = r.x + r.width  * 0.5;
      d.bbox.center.position.y = r.y + r.height * 0.5;
      d.bbox.center.theta = 0.0;
      d.bbox.size_x = r.width; d.bbox.size_y = r.height;

      vision::ObjectHypothesisWithPose hyp;
      hyp.hypothesis.class_id = class_name(c);
      hyp.hypothesis.score = s;
      d.results.push_back(hyp);
      out.detections.push_back(d);
    }

    pub_detections_->publish(out);
    if (publish_annotated_) {
      auto msg_img = cv_bridge::CvImage(msg->header, "bgr8", annotated).toImageMsg();
      pub_annotated_.publish(msg_img);
    }
  }

  std::string class_name(int c) const {
    if (c >= 0 && c < static_cast<int>(class_labels_.size()))
      return class_labels_[static_cast<size_t>(c)];
    return std::to_string(c);
  }

  // Params
  std::string input_topic_, annotated_topic_, detections_topic_;
  bool publish_annotated_{true};
  std::string engine_path_, input_binding_, output_binding_;
  double conf_th_{0.35}, iou_th_{0.5};
  int max_detections_{300};
  bool use_cuda_preproc_{false};
  std::vector<std::string> class_labels_;

  // ROS
  rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr sub_;
  rclcpp::Publisher<vision::Detection2DArray>::SharedPtr pub_detections_;
  image_transport::Publisher pub_annotated_;

  // TRT
  std::unique_ptr<nvinfer1::IRuntime> runtime_;
  std::unique_ptr<nvinfer1::ICudaEngine> engine_;
  std::unique_ptr<nvinfer1::IExecutionContext> context_;

  nvinfer1::Dims input_dims_{}, outDims_{};
  int netN_{1}, netC_{3}, netH_{640}, netW_{640};
  nvinfer1::DataType input_type_{nvinfer1::DataType::kFLOAT}, output_type_{nvinfer1::DataType::kFLOAT};

  cudaStream_t stream_{};
  void* dIn_{nullptr};
  void* dOut_{nullptr};

  size_t input_count_{0}, inBytes_{0};
  size_t outBytes_{0}; int outCount_{0};

  std::vector<OutBuf> outputs_;

  // Host buffers
  std::vector<float>    hostInputF_;
  std::vector<__half>   hostInputH_;
  std::vector<float>    hostOutputF_;
  std::vector<uint8_t>  hostOutputRaw_;
};

int main(int argc, char** argv) {
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<TrtDetectorNode>());
  rclcpp::shutdown();
  return 0;
}

#else  // SHB_USE_TRT

// Stub if TensorRT is disabled at build time
class TrtDetectorNode : public rclcpp::Node {
public:
  TrtDetectorNode() : rclcpp::Node("trt_detector_stub") {
    auto input_topic = declare_parameter<std::string>("input_topic", "/rgb_cam/image_raw");
    auto det_topic   = declare_parameter<std::string>("detections_topic", "/perception/detections");
    sub_ = create_subscription<sensor_msgs::msg::Image>(
      input_topic, rclcpp::SensorDataQoS(),
      std::bind(&TrtDetectorNode::on_image, this, std::placeholders::_1));
    pub_ = create_publisher<vision::Detection2DArray>(det_topic, 10);
    RCLCPP_WARN(get_logger(), "TensorRT not available. No detections.");
  }
private:
  void on_image(const sensor_msgs::msg::Image::ConstSharedPtr&) {
    static bool warned = false;
    if (!warned) { RCLCPP_WARN(get_logger(), "Received image but SHB_USE_TRT=0."); warned = true; }
  }
  rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr sub_;
  rclcpp::Publisher<vision::Detection2DArray>::SharedPtr pub_;
};

int main(int argc, char** argv) {
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<TrtDetectorNode>());
  rclcpp::shutdown();
  return 0;
}
#endif // SHB_USE_TRT
