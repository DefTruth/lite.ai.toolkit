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

  ortcv::FSANet *fsanet = new ortcv::FSANet(var_onnx_path, conv_onnx_path);
  cv::Mat roi = cv::imread(test_img_path);
  std::vector<float> euler_angles;

  // 1. 检测头部姿态
  fsanet->detect(roi, euler_angles);

  std::cout << "yaw: " << euler_angles[0]
            << " pitch: " << euler_angles[1]
            << " roll: " << euler_angles[2]
            << std::endl;

  delete fsanet;
}

int main(__unused int argc, __unused char *argv[]) {
  test_ortcv_fsanet();
  return 0;
}