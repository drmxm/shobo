#include <rclcpp/rclcpp.hpp>
#include <image_transport/image_transport.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <sensor_msgs/msg/camera_info.hpp>
#include <camera_info_manager/camera_info_manager.hpp>
#include <cv_bridge/cv_bridge.h>
#include <opencv2/opencv.hpp>
#include <sstream>

class CsiCamNode : public rclcpp::Node {
public:
  CsiCamNode() : Node("csi_cam_node") {
    int sensor_id = this->declare_parameter<int>("ir.sensor_id", 0);
    int width     = this->declare_parameter<int>("ir.width", 1280);
    int height    = this->declare_parameter<int>("ir.height", 720);
    int fps       = this->declare_parameter<int>("ir.fps", 30);
    frame_id_     = this->declare_parameter<std::string>("ir.frame_id", "ir_frame");
    std::string cname = this->declare_parameter<std::string>("ir.camera_name", "ir_imx219");
    std::string ciurl = this->declare_parameter<std::string>("ir.camera_info_url", "");

    pub_img_  = image_transport::create_publisher(this, "image_raw");
    pub_info_ = this->create_publisher<sensor_msgs::msg::CameraInfo>("camera_info", rclcpp::SensorDataQoS());
    cinfo_ = std::make_shared<camera_info_manager::CameraInfoManager>(this, cname, ciurl);

    std::ostringstream gst;
    gst << "nvarguscamerasrc sensor-id=" << sensor_id << " ! "
        << "video/x-raw(memory:NVMM),width="<<width<<",height="<<height<<",framerate="<<fps<<"/1,format=NV12 ! "
        << "queue max-size-buffers=1 leaky=downstream ! "
        << "nvvidconv ! video/x-raw,format=BGRx ! "
        << "videoconvert ! video/x-raw,format=BGR ! "
        << "appsink name=sink sync=false max-buffers=1 drop=true";

    cap_.open(gst.str(), cv::CAP_GSTREAMER);
    if (!cap_.isOpened()) throw std::runtime_error("Failed to open CSI via GStreamer");

    timer_ = this->create_wall_timer(std::chrono::milliseconds(1000/fps),
                                     std::bind(&CsiCamNode::tick, this));
    RCLCPP_INFO(get_logger(),"CSI IMX219 sensor-id=%d %dx%d@%dfps", sensor_id, width, height, fps);
  }

private:
  void tick(){
    cv::Mat bgr;
    if (!cap_.read(bgr)) return;
    if (bgr.empty()) {
      RCLCPP_WARN_THROTTLE(get_logger(), *this->get_clock(), 3000, "Received empty frame from %s", frame_id_.c_str());
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
};

int main(int argc, char** argv){
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<CsiCamNode>());
  rclcpp::shutdown();
  return 0;
}
