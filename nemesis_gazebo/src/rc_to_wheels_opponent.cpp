// For use to convert direct RC input to motor velocity commands (OPPONENT PUBLISHER)

#include <rclcpp/rclcpp.hpp>
#include <std_msgs/msg/float32_multi_array.hpp>
#include <std_msgs/msg/float64.hpp>
#include <algorithm>
#include <cmath>

int main(int argc, char** argv){
    rclcpp::init(argc, argv);
    auto n = rclcpp::Node::make_shared("rc_to_wheels_opponent");

    // 0-based indices into your channels array
    const int ch_left  = n->declare_parameter("ch_left",  0); // CH1
    const int ch_right = n->declare_parameter("ch_right", 1); // CH2
    const int ch_weap = n->declare_parameter("ch_weap", 2); // CH3

    // RC input is assumed normalized to [-1, 1]
    const double w_max = n->declare_parameter("w_max", 150);                // [rad/s] max wheel speed
    const double w_weap_max = n->declare_parameter("w_weap_max", 4000);              // [rad/s] max weapon speed
    const double db    = n->declare_parameter("deadband", 0.05);  // unitless
    const bool inv_l   = n->declare_parameter("invert_left",  false);
    const bool inv_r   = n->declare_parameter("invert_right", true);
    const bool inv_weap = n->declare_parameter("invert_weap", false);

    auto pubL = n->create_publisher<std_msgs::msg::Float64>("/opponent/left_wheels/cmd_vel", 10);
    auto pubR = n->create_publisher<std_msgs::msg::Float64>("/opponent/right_wheels/cmd_vel", 10);
    auto pubL_f = n->create_publisher<std_msgs::msg::Float64>("/opponent/left_wheels/cmd_vel_f", 10);
    auto pubR_f = n->create_publisher<std_msgs::msg::Float64>("/opponent/right_wheels/cmd_vel_f", 10);
    auto pubWeap = n->create_publisher<std_msgs::msg::Float64>("/opponent/weapon_motor/cmd_vel", 10);

    auto sub = n->create_subscription<std_msgs::msg::Float32MultiArray>(
        "/radiomaster/channels", 10,
        [&](std_msgs::msg::Float32MultiArray::SharedPtr m){
        if ((int)m->data.size() <= std::max(ch_left, ch_right)) return;

        double l = m->data[ch_left];
        double r = m->data[ch_right];
        double w = m->data[ch_weap];

        if (inv_l) l = -l;
        if (inv_r) r = -r;
        if (inv_weap) w = -w;

        if (std::abs(l) < db) l = 0.0;
        if (std::abs(r) < db) r = 0.0;
        if (std::abs(w) < db) w = 0.0;

        l = std::clamp(l, -1.0, 1.0);
        r = std::clamp(r, -1.0, 1.0);
        w = 0.5 * (w + 1.0);
        w = std::clamp(w, 0.0, 1.0);

        std_msgs::msg::Float64 wl, wr, wlf, wrf, ww;
        wl.data = w_max * l;
        wr.data = w_max * r;
        wlf.data = wl.data * 1.2143; // Some constants for vert (sus)
        wrf.data = wr.data * 1.2143;
        ww.data = w_weap_max * w;

        pubL->publish(wl);
        pubR->publish(wr);
        pubL_f->publish(wlf);
        pubR_f->publish(wrf);
        pubWeap->publish(ww);
        }
    );

    rclcpp::spin(n);
    rclcpp::shutdown();
    return 0;
}
