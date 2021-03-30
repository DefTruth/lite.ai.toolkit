//
// Created by YanJun Qiu on 2021/3/21.
//

#include <iostream>
#include <vector>

#include "ort/cv/fsanet.h"
#include "ort/core/ort_utils.h"

static void test_ortcv_fsanet() {

  std::string var_onnx_path = "../../../hub/onnx/cv/fsanet-var.onnx";
  std::string conv_onnx_path = "../../../hub/onnx/cv/fsanet-1x1.onnx";
  std::string test_img_path = "../../../examples/ort/resources/test_ortcv_fsanet.jpg";
  std::string save_img_path = "../../../logs/test_ortcv_fsanet.jpg";

  ortcv::FSANet *var_fsanet = new ortcv::FSANet(var_onnx_path);
  ortcv::FSANet *conv_fsanet = new ortcv::FSANet(conv_onnx_path);
  cv::Mat roi = cv::imread(test_img_path);
  ortcv::types::EulerAngles var_euler_angles, conv_euler_angles;

  // 1. detect euler angles.
  var_fsanet->detect(roi, var_euler_angles);
  conv_fsanet->detect(roi, conv_euler_angles);

  ortcv::types::EulerAngles euler_angles;

  euler_angles.yaw = (var_euler_angles.yaw + conv_euler_angles.yaw) / 2.0f;
  euler_angles.pitch = (var_euler_angles.pitch + conv_euler_angles.pitch) / 2.0f;
  euler_angles.roll = (var_euler_angles.roll + conv_euler_angles.roll) / 2.0f;

  // 2. draw euler angles.
  ortcv::utils::draw_axis_inplace(roi, euler_angles);

  cv::imwrite(save_img_path, roi);

  std::cout << "yaw: " << euler_angles.yaw
            << " pitch: " << euler_angles.pitch
            << " roll: " << euler_angles.roll << std::endl;

  delete var_fsanet; delete conv_fsanet;
}

int main(__unused int argc, __unused char *argv[]) {
  test_ortcv_fsanet();
  return 0;
}