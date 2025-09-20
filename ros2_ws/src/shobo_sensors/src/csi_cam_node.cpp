#include <rclcpp/rclcpp.hpp>
#include <image_transport/image_transport.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <sensor_msgs/msg/camera_info.hpp>
#include <camera_info_manager/camera_info_manager.hpp>
#include <cv_bridge/cv_bridge.h>
#include <opencv2/opencv.hpp>
#include <algorithm>
#include <sstream>

class CsiCamNode : public rclcpp::Node {
public:
  CsiCamNode() : Node("csi_cam_node") {
    sensor_id_ = this->declare_parameter<int>("ir.sensor_id", 0);
    width_     = this->declare_parameter<int>("ir.width", 1280);
    height_    = this->declare_parameter<int>("ir.height", 720);
    fps_       = this->declare_parameter<int>("ir.fps", 30);
    frame_id_  = this->declare_parameter<std::string>("ir.frame_id", "ir_frame");
    std::string cname = this->declare_parameter<std::string>("ir.camera_name", "ir_imx219");
    std::string ciurl = this->declare_parameter<std::string>("ir.camera_info_url", "");

    pub_img_  = image_transport::create_publisher(this, "image_raw");
    pub_info_ = this->create_publisher<sensor_msgs::msg::CameraInfo>("camera_info", rclcpp::SensorDataQoS());
    cinfo_ = std::make_shared<camera_info_manager::CameraInfoManager>(this, cname, ciurl);

    if (!open_camera()) {
      throw std::runtime_error("Failed to open CSI via GStreamer");
    }

    timer_ = this->create_wall_timer(std::chrono::milliseconds(std::max(1, 1000 / std::max(1, fps_))),
                                     std::bind(&CsiCamNode::tick, this));
    RCLCPP_INFO(get_logger(),"CSI IMX219 sensor-id=%d %dx%d@%dfps", sensor_id_, width_, height_, fps_);
  }

private:
  bool open_camera() {
    cap_.release();
    std::ostringstream gst;
    gst << "nvarguscamerasrc sensor-id=" << sensor_id_ << " ! "
        << "video/x-raw(memory:NVMM),width=" << width_ << ",height=" << height_
        << ",framerate=" << fps_ << "/1,format=NV12 ! "
        << "nvvidconv ! video/x-raw,format=BGRx ! "
        << "videoconvert ! video/x-raw,format=BGR ! "
        << "appsink drop=true sync=false max-buffers=1";

    if (!cap_.open(gst.str(), cv::CAP_GSTREAMER)) {
      RCLCPP_WARN(get_logger(), "Failed to open CSI sensor %d", sensor_id_);
      return false;
    }

    return true;
  }

  void tick(){
    if (!cap_.isOpened()) {
      if (open_camera()) {
        RCLCPP_INFO(get_logger(), "Re-opened CSI sensor %d", sensor_id_);
      }
      return;
    }

    cv::Mat bgr;
    try {
      if (!cap_.read(bgr) || bgr.empty()) {
        RCLCPP_WARN_THROTTLE(get_logger(), *this->get_clock(), 3000, "Empty frame from CSI %d", sensor_id_);
        cap_.release();
        return;
      }
    } catch (const cv::Exception &e) {
      RCLCPP_WARN(get_logger(), "CSI capture error: %s", e.what());
      cap_.release();
      return;
    }

    auto stamp = this->now();
    auto msg = cv_bridge::CvImage(std_msgs::msg::Header(), "bgr8", bgr).toImageMsg();
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
  int sensor_id_{0};
  int width_{0};
  int height_{0};
  int fps_{0};
};

int main(int argc, char** argv){
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<CsiCamNode>());
  rclcpp::shutdown();
  return 0;
}
