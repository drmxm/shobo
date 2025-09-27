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

#include "shobo_detectors/kernels.hpp"

#if SHB_USE_TRT

#include <NvInfer.h>
#include <cuda_runtime.h>

#include <algorithm>
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
  void log(Severity severity, const char* message) noexcept override {
    if (severity <= Severity::kWARNING) {
      RCLCPP_INFO(rclcpp::get_logger("trt_detector"), "[TRT] %s", message);
    }
  }
};

Logger gLogger;

#define CHECK_CUDA(cmd)                                                                             \
  do {                                                                                              \
    cudaError_t e = (cmd);                                                                          \
    if (e != cudaSuccess) {                                                                         \
      throw std::runtime_error(std::string("CUDA error: ") + cudaGetErrorString(e));               \
    }                                                                                               \
  } while (0)

float intersection_over_union(const cv::Rect& a, const cv::Rect& b) {
  const int x1 = std::max(a.x, b.x);
  const int y1 = std::max(a.y, b.y);
  const int x2 = std::min(a.x + a.width, b.x + b.width);
  const int y2 = std::min(a.y + a.height, b.y + b.height);

  const int interW = std::max(0, x2 - x1);
  const int interH = std::max(0, y2 - y1);
  const int interArea = interW * interH;
  if (interArea <= 0) {
    return 0.f;
  }

  const int unionArea = a.area() + b.area() - interArea;
  if (unionArea <= 0) {
    return 0.f;
  }

  return static_cast<float>(interArea) / static_cast<float>(unionArea);
}

void nms_indices(const std::vector<cv::Rect>& boxes,
                 const std::vector<float>& scores,
                 float iou_threshold,
                 std::vector<int>& keep) {
  const size_t count = boxes.size();
  if (count == 0) {
    keep.clear();
    return;
  }

  std::vector<int> order(count);
  std::iota(order.begin(), order.end(), 0);
  std::sort(order.begin(), order.end(), [&](int lhs, int rhs) {
    return scores[static_cast<size_t>(lhs)] > scores[static_cast<size_t>(rhs)];
  });

  std::vector<char> suppressed(count, 0);
  keep.clear();
  keep.reserve(count);

  for (size_t i = 0; i < count; ++i) {
    const int idx = order[i];
    if (suppressed[static_cast<size_t>(idx)]) {
      continue;
    }
    keep.push_back(idx);

    for (size_t j = i + 1; j < count; ++j) {
      const int nextIdx = order[j];
      if (suppressed[static_cast<size_t>(nextIdx)]) {
        continue;
      }

      const float iou = intersection_over_union(boxes[static_cast<size_t>(idx)],
                                                 boxes[static_cast<size_t>(nextIdx)]);
      if (iou >= iou_threshold) {
        suppressed[static_cast<size_t>(nextIdx)] = 1;
      }
    }
  }
}

}  // namespace

class TrtDetectorNode : public rclcpp::Node {
public:
  TrtDetectorNode() : rclcpp::Node("trt_detector") {
    rclcpp::SensorDataQoS qos;
    qos.keep_last(5);
    qos.best_effort();

    input_topic_ = this->declare_parameter<std::string>("input_topic", "/sensors/rgb/image_raw");
    annotated_topic_ = this->declare_parameter<std::string>("annotated_topic", "/perception/annotated");
    detections_topic_ = this->declare_parameter<std::string>("detections_topic", "/perception/detections");
    publish_annotated_ = this->declare_parameter<bool>("publish_annotated", true);
    engine_path_ = this->declare_parameter<std::string>("engine_path", "/work/ros2_ws/models/yolov8n.engine");
    input_binding_ = this->declare_parameter<std::string>("input_binding", "images");
    output_binding_ = this->declare_parameter<std::string>("output_binding", "output0");
    conf_th_ = this->declare_parameter<double>("conf_th", 0.35);
    iou_th_ = this->declare_parameter<double>("iou_th", 0.50);
    max_detections_ = this->declare_parameter<int>("max_detections", 300);
    use_cuda_preproc_ = this->declare_parameter<bool>("use_cuda_preproc", true);
    class_labels_ = this->declare_parameter<std::vector<std::string>>(
      "class_labels", std::vector<std::string>());

    load_engine();

    sub_ = this->create_subscription<sensor_msgs::msg::Image>(
      input_topic_, qos, std::bind(&TrtDetectorNode::on_image, this, std::placeholders::_1));

    if (publish_annotated_) {
      pub_annotated_ = image_transport::create_publisher(this, annotated_topic_);
    }

    pub_detections_ = this->create_publisher<vision::Detection2DArray>(detections_topic_, qos);

    RCLCPP_INFO(get_logger(), "TensorRT detector ready: input=%s output=%s dims=[%d,%d,%d]",
                input_binding_.c_str(), output_binding_.c_str(), netC_, netH_, netW_);
  }

  ~TrtDetectorNode() override {
    try {
      if (stream_) {
        cudaStreamDestroy(stream_);
      }
      if (dIn_) {
        cudaFree(dIn_);
      }
      if (dOut_) {
        cudaFree(dOut_);
      }
    } catch (...) {
      // suppress exceptions in dtor
    }
  }

private:
  void load_engine() {
    std::ifstream file(engine_path_, std::ios::binary);
    if (!file) {
      throw std::runtime_error("TensorRT engine not found at " + engine_path_);
    }

    file.seekg(0, std::ios::end);
    const size_t size = static_cast<size_t>(file.tellg());
    file.seekg(0, std::ios::beg);
    std::vector<char> buffer(size);
    file.read(buffer.data(), size);

    runtime_.reset(nvinfer1::createInferRuntime(gLogger));
    if (!runtime_) {
      throw std::runtime_error("Failed to create TensorRT runtime");
    }

    engine_.reset(runtime_->deserializeCudaEngine(buffer.data(), size));
    if (!engine_) {
      throw std::runtime_error("Failed to deserialize engine");
    }

    context_.reset(engine_->createExecutionContext());
    if (!context_) {
      throw std::runtime_error("Failed to create execution context");
    }

    const int nbIO = engine_->getNbIOTensors();
    inputs_.clear();
    outputs_.clear();
    for (int i = 0; i < nbIO; ++i) {
      const char* name = engine_->getIOTensorName(i);
      const auto mode = engine_->getTensorIOMode(name);
      if (mode == nvinfer1::TensorIOMode::kINPUT) {
        inputs_.emplace_back(name);
      } else if (mode == nvinfer1::TensorIOMode::kOUTPUT) {
        outputs_.emplace_back(name);
      }
    }

    if (std::find(inputs_.begin(), inputs_.end(), input_binding_) == inputs_.end()) {
      if (!inputs_.empty()) {
        RCLCPP_WARN(get_logger(), "Input binding '%s' not found. Using '%s' instead.",
                    input_binding_.c_str(), inputs_.front().c_str());
        input_binding_ = inputs_.front();
      } else {
        throw std::runtime_error("Engine has no input tensors");
      }
    }

    if (std::find(outputs_.begin(), outputs_.end(), output_binding_) == outputs_.end()) {
      if (!outputs_.empty()) {
        RCLCPP_WARN(get_logger(), "Output binding '%s' not found. Using '%s' instead.",
                    output_binding_.c_str(), outputs_.front().c_str());
        output_binding_ = outputs_.front();
      } else {
        throw std::runtime_error("Engine has no output tensors");
      }
    }

    nvinfer1::Dims inDims = engine_->getTensorShape(input_binding_.c_str());
    if (inDims.nbDims != 4) {
      throw std::runtime_error("Expected NCHW input tensor");
    }
    netN_ = inDims.d[0];
    netC_ = inDims.d[1];
    netH_ = inDims.d[2];
    netW_ = inDims.d[3];
    if (netN_ != 1) {
      throw std::runtime_error("Only batch size 1 engines are supported");
    }

    inBytes_ = static_cast<size_t>(netC_) * netH_ * netW_ * sizeof(float);

    outDims_ = engine_->getTensorShape(output_binding_.c_str());
    if (outDims_.nbDims < 3) {
      throw std::runtime_error("Unexpected output dims; expected [1, attrs, boxes]");
    }

    outCount_ = 1;
    for (int i = 0; i < outDims_.nbDims; ++i) {
      outCount_ *= outDims_.d[i];
    }
    outBytes_ = static_cast<size_t>(outCount_) * sizeof(float);

    CHECK_CUDA(cudaStreamCreate(&stream_));
    CHECK_CUDA(cudaMalloc(&dIn_, inBytes_));
    CHECK_CUDA(cudaMalloc(&dOut_, outBytes_));

    hostInput_.resize(inBytes_ / sizeof(float));
    hostOutput_.resize(outCount_);
  }

  void on_image(const sensor_msgs::msg::Image::ConstSharedPtr& msg) {
    cv_bridge::CvImageConstPtr cv_ptr;
    try {
      cv_ptr = cv_bridge::toCvShare(msg, "bgr8");
    } catch (const cv_bridge::Exception& e) {
      RCLCPP_WARN_THROTTLE(get_logger(), *this->get_clock(), 2000,
                           "cv_bridge failed: %s", e.what());
      return;
    }

    const cv::Mat& frame = cv_ptr->image;
    if (frame.empty()) {
      RCLCPP_WARN_THROTTLE(get_logger(), *this->get_clock(), 2000, "Empty frame received");
      return;
    }

    float scale = std::min(static_cast<float>(netW_) / frame.cols,
                           static_cast<float>(netH_) / frame.rows);
    int scaledW = static_cast<int>(std::round(frame.cols * scale));
    int scaledH = static_cast<int>(std::round(frame.rows * scale));
    int padX = (netW_ - scaledW) / 2;
    int padY = (netH_ - scaledH) / 2;

    if (!preprocess(frame, scale, padX, padY)) {
      return;
    }

    context_->setTensorAddress(input_binding_.c_str(), dIn_);
    context_->setTensorAddress(output_binding_.c_str(), dOut_);

    if (!context_->enqueueV3(stream_)) {
      RCLCPP_WARN_THROTTLE(get_logger(), *this->get_clock(), 2000, "TensorRT enqueue failed");
      return;
    }

    CHECK_CUDA(cudaMemcpyAsync(hostOutput_.data(), dOut_, outBytes_, cudaMemcpyDeviceToHost, stream_));
    CHECK_CUDA(cudaStreamSynchronize(stream_));

    publish_detections(msg, frame, scale, padX, padY);
  }

  bool preprocess(const cv::Mat& frame, float scale, int padX, int padY) {
    if (use_cuda_preproc_) {
      try {
        const unsigned char* src = frame.data;
        shobo::bgr_to_nchw_norm(src, frame.cols, frame.rows, static_cast<int>(frame.step),
                                reinterpret_cast<float*>(dIn_), netW_, netH_,
                                scale, scale, padX, padY, stream_);
        CHECK_CUDA(cudaGetLastError());
      } catch (const std::exception& e) {
        RCLCPP_WARN(get_logger(), "CUDA preprocess failed: %s. Falling back to CPU.", e.what());
        use_cuda_preproc_ = false;
      }
    }

    if (!use_cuda_preproc_) {
      // CPU letterbox → float32 NCHW / 0..1 normalization
      cv::Mat resized;
      cv::resize(frame, resized, cv::Size(), scale, scale, cv::INTER_LINEAR);

      cv::Mat canvas(netH_, netW_, CV_8UC3, cv::Scalar(114, 114, 114));
      resized.copyTo(canvas(cv::Rect(padX, padY, resized.cols, resized.rows)));

      cv::Mat canvasFloat;
      canvas.convertTo(canvasFloat, CV_32FC3, 1.f / 255.f);

      const int channelSize = netH_ * netW_;
      std::vector<cv::Mat> channels(3);
      cv::split(canvasFloat, channels);

      for (int c = 0; c < 3; ++c) {
        std::memcpy(hostInput_.data() + c * channelSize, channels[c].data,
                    channelSize * sizeof(float));
      }

      CHECK_CUDA(cudaMemcpyAsync(dIn_, hostInput_.data(), inBytes_, cudaMemcpyHostToDevice, stream_));
      CHECK_CUDA(cudaStreamSynchronize(stream_));
    }

    return true;
  }

  void publish_detections(const sensor_msgs::msg::Image::ConstSharedPtr& msg,
                          const cv::Mat& frame, float scale, int padX, int padY) {
    const int nAttr = outDims_.nbDims > 1 ? outDims_.d[1] : 0;
    const int nBoxes = outDims_.nbDims > 2 ? outDims_.d[2] : 0;

    auto accessor = [&](int attr, int box) -> float {
      return hostOutput_[attr * nBoxes + box];
    };

    std::vector<cv::Rect> boxes;
    std::vector<int> classes;
    std::vector<float> scores;

    boxes.reserve(nBoxes);
    classes.reserve(nBoxes);
    scores.reserve(nBoxes);

    for (int b = 0; b < nBoxes; ++b) {
      float obj = accessor(4, b);
      int bestClass = -1;
      float bestProb = 0.f;
      for (int c = 5; c < nAttr; ++c) {
        float prob = accessor(c, b);
        if (prob > bestProb) {
          bestProb = prob;
          bestClass = c - 5;
        }
      }

      float conf = obj * bestProb;
      if (conf < static_cast<float>(conf_th_)) {
        continue;
      }

      float cx = accessor(0, b);
      float cy = accessor(1, b);
      float w = accessor(2, b);
      float h = accessor(3, b);

      float left = (cx - w / 2.f - padX) / scale;
      float top = (cy - h / 2.f - padY) / scale;
      float width = w / scale;
      float height = h / scale;

      int x = std::max(0, static_cast<int>(std::round(left)));
      int y = std::max(0, static_cast<int>(std::round(top)));
      int ww = std::min(frame.cols - x, static_cast<int>(std::round(width)));
      int hh = std::min(frame.rows - y, static_cast<int>(std::round(height)));
      if (ww <= 0 || hh <= 0) {
        continue;
      }

      boxes.emplace_back(x, y, ww, hh);
      classes.emplace_back(bestClass);
      scores.emplace_back(conf);
    }

    std::vector<int> keep;
    nms_indices(boxes, scores, static_cast<float>(iou_th_), keep);

    if (max_detections_ > 0 && static_cast<int>(keep.size()) > max_detections_) {
      keep.resize(static_cast<size_t>(max_detections_));
    }

    vision::Detection2DArray detections;
    detections.header = msg->header;

    cv::Mat annotated = frame.clone();

    for (int idx : keep) {
      const auto& rect = boxes[idx];
      const float score = scores[idx];
      const int cls = classes[idx];

      if (publish_annotated_) {
        cv::rectangle(annotated, rect, cv::Scalar(0, 255, 0), 2);
        cv::putText(annotated,
                    format_label(cls, score),
                    cv::Point(rect.x, std::max(0, rect.y - 5)),
                    cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 255, 0), 1);
      }

      vision::Detection2D det;
      det.bbox.center.position.x = rect.x + rect.width * 0.5;
      det.bbox.center.position.y = rect.y + rect.height * 0.5;
      det.bbox.center.theta = 0.0;
      det.bbox.size_x = rect.width;
      det.bbox.size_y = rect.height;

      vision::ObjectHypothesisWithPose hyp;
      hyp.hypothesis.class_id = class_name(cls);
      hyp.hypothesis.score = score;
      det.results.push_back(hyp);
      detections.detections.push_back(det);
    }

    pub_detections_->publish(detections);

    if (publish_annotated_ && pub_annotated_) {
      auto annotated_msg = cv_bridge::CvImage(msg->header, "bgr8", annotated).toImageMsg();
      pub_annotated_.publish(annotated_msg);
    }
  }

  std::string class_name(int cls) const {
    if (cls >= 0 && cls < static_cast<int>(class_labels_.size())) {
      return class_labels_[static_cast<size_t>(cls)];
    }
    return std::to_string(cls);
  }

  std::string format_label(int cls, float score) const {
    std::ostringstream oss;
    oss.setf(std::ios::fixed, std::ios::floatfield);
    oss.precision(2);
    oss << class_name(cls) << " " << score;
    return oss.str();
  }

  // Parameters / configuration
  std::string input_topic_;
  std::string annotated_topic_;
  std::string detections_topic_;
  bool publish_annotated_{true};
  std::string engine_path_;
  std::string input_binding_;
  std::string output_binding_;
  double conf_th_{0.35};
  double iou_th_{0.5};
  int max_detections_{300};
  bool use_cuda_preproc_{false};
  std::vector<std::string> class_labels_;

  // ROS interfaces
  rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr sub_;
  rclcpp::Publisher<vision::Detection2DArray>::SharedPtr pub_detections_;
  image_transport::Publisher pub_annotated_;

  // TensorRT objects
  std::unique_ptr<nvinfer1::IRuntime> runtime_;
  std::unique_ptr<nvinfer1::ICudaEngine> engine_;
  std::unique_ptr<nvinfer1::IExecutionContext> context_;
  std::vector<std::string> inputs_;
  std::vector<std::string> outputs_;
  nvinfer1::Dims outDims_{};
  int netN_{1};
  int netC_{3};
  int netH_{640};
  int netW_{640};
  size_t inBytes_{0};
  size_t outBytes_{0};
  int outCount_{0};

  cudaStream_t stream_{};
  void* dIn_{nullptr};
  void* dOut_{nullptr};
  std::vector<float> hostInput_;
  std::vector<float> hostOutput_;
};

int main(int argc, char** argv) {
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<TrtDetectorNode>());
  rclcpp::shutdown();
  return 0;
}

#else  // SHB_USE_TRT

class TrtDetectorNode : public rclcpp::Node {
public:
  TrtDetectorNode() : rclcpp::Node("trt_detector_stub") {
    input_topic_ = this->declare_parameter<std::string>("input_topic", "/sensors/rgb/image_raw");
    detections_topic_ = this->declare_parameter<std::string>("detections_topic", "/perception/detections");

    sub_ = this->create_subscription<sensor_msgs::msg::Image>(
      input_topic_, rclcpp::SensorDataQoS(),
      std::bind(&TrtDetectorNode::on_image, this, std::placeholders::_1));

    pub_detections_ = this->create_publisher<vision::Detection2DArray>(detections_topic_, 10);

    RCLCPP_WARN(get_logger(), "TensorRT not available. Detector node will not produce detections.");
  }

private:
  void on_image(const sensor_msgs::msg::Image::ConstSharedPtr& msg) {
    (void)msg;
    if (!warned_) {
      RCLCPP_WARN(get_logger(), "Received image but TensorRT is not enabled (SHB_USE_TRT=0)." );
      warned_ = true;
    }
  }

  std::string input_topic_;
  std::string detections_topic_;
  rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr sub_;
  rclcpp::Publisher<vision::Detection2DArray>::SharedPtr pub_detections_;
  bool warned_{false};
};

int main(int argc, char** argv) {
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<TrtDetectorNode>());
  rclcpp::shutdown();
  return 0;
}

#endif  // SHB_USE_TRT
