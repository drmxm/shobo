#include <rclcpp/rclcpp.hpp>
#include <image_transport/image_transport.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <sensor_msgs/msg/camera_info.hpp>
#include <camera_info_manager/camera_info_manager.hpp>
#include <cv_bridge/cv_bridge.h>
#include <opencv2/opencv.hpp>

class UvcCamNode : public rclcpp::Node {
public:
  UvcCamNode() : Node("uvc_cam_node") {
    std::string device = this->declare_parameter<std::string>("rgb.device", "/dev/video1");
    int width  = this->declare_parameter<int>("rgb.width", 1280);
    int height = this->declare_parameter<int>("rgb.height", 720);
    int fps    = this->declare_parameter<int>("rgb.fps", 30);
    frame_id_  = this->declare_parameter<std::string>("rgb.frame_id", "rgb_frame");
    std::string cname = this->declare_parameter<std::string>("rgb.camera_name", "rgb_uvc");
    std::string ciurl = this->declare_parameter<std::string>("rgb.camera_info_url", "");

    it_ = std::make_shared<image_transport::ImageTransport>(shared_from_this());
    pub_img_ = it_->advertise("image_raw", 1);
    pub_info_ = this->create_publisher<sensor_msgs::msg::CameraInfo>("camera_info", rclcpp::SensorDataQoS());
    cinfo_ = std::make_shared<camera_info_manager::CameraInfoManager>(this, cname, ciurl);

    cap_.open(device, cv::CAP_V4L2);
    if (!cap_.isOpened()) throw std::runtime_error("Failed to open " + device);
    cap_.set(cv::CAP_PROP_FRAME_WIDTH,  width);
    cap_.set(cv::CAP_PROP_FRAME_HEIGHT, height);
    cap_.set(cv::CAP_PROP_FPS, fps);

    timer_ = this->create_wall_timer(std::chrono::milliseconds(1000/fps),
                                     std::bind(&UvcCamNode::tick, this));
    RCLCPP_INFO(get_logger(), "UVC %s @%dx%d@%dfps", device.c_str(), width, height, fps);
  }

private:
  void tick(){
    cv::Mat frame;
    if (!cap_.read(frame)) return;

    auto stamp = this->now();
    auto msg = cv_bridge::CvImage(std_msgs::msg::Header(), "bgr8", frame).toImageMsg();
    msg->header.stamp = stamp;
    msg->header.frame_id = frame_id_;

    auto info = cinfo_->getCameraInfo();
    info.header = msg->header;

    pub_img_.publish(msg);
    pub_info_->publish(info);
  }

  std::shared_ptr<image_transport::ImageTransport> it_;
  image_transport::Publisher pub_img_;
  rclcpp::Publisher<sensor_msgs::msg::CameraInfo>::SharedPtr pub_info_;
  std::shared_ptr<camera_info_manager::CameraInfoManager> cinfo_;
  cv::VideoCapture cap_;
  rclcpp::TimerBase::SharedPtr timer_;
  std::string frame_id_;
};

int main(int argc, char** argv){
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<UvcCamNode>());
  rclcpp::shutdown();
  return 0;
}
