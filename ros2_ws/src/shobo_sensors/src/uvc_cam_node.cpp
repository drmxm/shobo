#include <rclcpp/rclcpp.hpp>
#include <image_transport/image_transport.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <sensor_msgs/msg/camera_info.hpp>
#include <camera_info_manager/camera_info_manager.hpp>
#include <cv_bridge/cv_bridge.h>
#include <opencv2/opencv.hpp>
#include <algorithm>
#include <sstream>

class UvcCamNode : public rclcpp::Node {
public:
  UvcCamNode() : Node("uvc_cam_node") {
    device_ = this->declare_parameter<std::string>("rgb.device", "/dev/video1");
    width_  = this->declare_parameter<int>("rgb.width", 1280);
    height_ = this->declare_parameter<int>("rgb.height", 720);
    fps_    = this->declare_parameter<int>("rgb.fps", 30);
    frame_id_  = this->declare_parameter<std::string>("rgb.frame_id", "rgb_frame");
    std::string cname = this->declare_parameter<std::string>("rgb.camera_name", "rgb_uvc");
    std::string ciurl = this->declare_parameter<std::string>("rgb.camera_info_url", "");

    pub_img_ = image_transport::create_publisher(this, "image_raw");
    pub_info_ = this->create_publisher<sensor_msgs::msg::CameraInfo>("camera_info", rclcpp::SensorDataQoS());
    cinfo_ = std::make_shared<camera_info_manager::CameraInfoManager>(this, cname, ciurl);

    if (!open_camera()) {
      throw std::runtime_error("Failed to open " + device_);
    }

    timer_ = this->create_wall_timer(std::chrono::milliseconds(std::max(1, 1000 / std::max(1, fps_))),
                                     std::bind(&UvcCamNode::tick, this));
    RCLCPP_INFO(get_logger(), "UVC %s @%dx%d@%dfps (%s)", device_.c_str(), width_, height_, fps_,
                using_gstreamer_ ? "GStreamer" : "V4L2");
  }

private:
  bool open_camera(bool relax = true) {
    cap_.release();

    auto build_pipeline = [&](bool include_fps) {
      std::ostringstream pipeline;
      pipeline << "v4l2src device=" << device_;
      pipeline << " ! video/x-raw,width=" << width_ << ",height=" << height_;
      if (include_fps && fps_ > 0) {
        pipeline << ",framerate=" << fps_ << "/1";
      }
      pipeline << " ! videoconvert ! video/x-raw,format=BGR ! appsink drop=true sync=false max-buffers=1";
      return pipeline.str();
    };

    if (!force_v4l2_) {
      if (cap_.open(build_pipeline(true), cv::CAP_GSTREAMER)) {
        using_gstreamer_ = true;
        gst_failures_ = 0;
        return true;
      }

      if (fps_ > 0 && cap_.open(build_pipeline(false), cv::CAP_GSTREAMER)) {
        RCLCPP_WARN(get_logger(), "Opened %s without enforcing FPS caps", device_.c_str());
        using_gstreamer_ = true;
        gst_failures_ = 0;
        return true;
      }

      gst_failures_++;
      if (gst_failures_ >= 3) {
        force_v4l2_ = true;
        RCLCPP_WARN(get_logger(), "Disabling GStreamer path for %s after %d failures", device_.c_str(), gst_failures_);
      }
    }

    if (!relax) {
      return false;
    }

    cap_.release();
    if (cap_.open(device_, cv::CAP_V4L2)) {
      using_gstreamer_ = false;
      gst_failures_ = 0;
      cap_.set(cv::CAP_PROP_FRAME_WIDTH,  width_);
      cap_.set(cv::CAP_PROP_FRAME_HEIGHT, height_);
      if (fps_ > 0) cap_.set(cv::CAP_PROP_FPS, fps_);
      return true;
    }

    return false;
  }

  void tick(){
    if (!cap_.isOpened()) {
      if (open_camera()) {
        RCLCPP_INFO(get_logger(), "Re-opened %s (%s)", device_.c_str(), using_gstreamer_ ? "GStreamer" : "V4L2");
      } else {
        RCLCPP_WARN_THROTTLE(get_logger(), *this->get_clock(), 3000, "Camera %s unavailable", device_.c_str());
      }
      return;
    }

    cv::Mat frame;
    try {
      if (!cap_.read(frame) || frame.empty()) {
        RCLCPP_WARN_THROTTLE(get_logger(), *this->get_clock(), 3000, "Empty frame from %s", device_.c_str());
        cap_.release();
        return;
      }
    } catch (const cv::Exception &e) {
      RCLCPP_WARN(get_logger(), "Capture error on %s: %s", device_.c_str(), e.what());
      cap_.release();
      return;
    }

    auto stamp = this->now();
    auto msg = cv_bridge::CvImage(std_msgs::msg::Header(), "bgr8", frame).toImageMsg();
    msg->header.stamp = stamp;
    msg->header.frame_id = frame_id_;

    auto info = cinfo_->getCameraInfo();
    info.header = msg->header;

    pub_img_.publish(msg);
    pub_info_->publish(info);
  }

  image_transport::Publisher pub_img_;
  rclcpp::Publisher<sensor_msgs::msg::CameraInfo>::SharedPtr pub_info_;
  std::shared_ptr<camera_info_manager::CameraInfoManager> cinfo_;
  cv::VideoCapture cap_;
  rclcpp::TimerBase::SharedPtr timer_;
  std::string frame_id_;
  std::string device_;
  int width_{0};
  int height_{0};
  int fps_{0};
  bool using_gstreamer_{false};
  int gst_failures_{0};
  bool force_v4l2_{false};
};

int main(int argc, char** argv){
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<UvcCamNode>());
  rclcpp::shutdown();
  return 0;
}
