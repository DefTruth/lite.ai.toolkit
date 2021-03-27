//
// Created by YanJun Qiu on 2021/3/21.
//

#include <iostream>
#include <vector>

#include "ort/cv/fsanet.h"


static void test_ortcv_fsanet() {

  std::string var_onnx_path = "../../../hub/onnx/cv/fsanet-var.onnx";
  std::string conv_onnx_path = "../../../hub/onnx/cv/fsanet-1x1.onnx";
  std::string test_img_path = "../../../examples/ort/resources/test_ortcv_fsanet.jpg";
  std::string save_img_path = "../../../logs/test_ortcv_fsanet.jpg";

  ortcv::FSANet *fsanet = new ortcv::FSANet(var_onnx_path, conv_onnx_path);
  cv::Mat roi = cv::imread(test_img_path);
  std::vector<float> euler_angles;

  // 1. 检测头部姿态
  fsanet->detect(roi, euler_angles);

  const float yaw = euler_angles.at(0);
  const float pitch = euler_angles.at(1);
  const float roll = euler_angles.at(2);

  // 2. 绘制欧拉角
  // cv::Mat out_img = ortcv::FSANet::draw_axis(roi, yaw, pitch, roll);
  ortcv::FSANet::draw_axis_inplace(roi, yaw, pitch, roll);

  cv::imwrite(save_img_path, roi);

  std::cout << "yaw: " << yaw << " pitch: " << pitch << " roll: " << roll << std::endl;

  delete fsanet;
}

int main(__unused int argc, __unused char *argv[]) {
  test_ortcv_fsanet();
  return 0;
}