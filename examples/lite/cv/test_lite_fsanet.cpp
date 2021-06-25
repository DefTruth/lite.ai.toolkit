//
// Created by DefTruth on 2021/6/24.
//

#include "lite/lite.h"

static void test_default()
{
  std::string var_onnx_path = "../../../hub/onnx/cv/fsanet-var.onnx";
  std::string conv_onnx_path = "../../../hub/onnx/cv/fsanet-1x1.onnx";
  std::string test_img_path = "../../../examples/lite/resources/test_lite_fsanet.jpg";
  std::string save_img_path = "../../../logs/test_lite_fsanet.jpg";

  lite::cv::face::FSANet *var_fsanet = new lite::cv::face::FSANet(var_onnx_path);
  lite::cv::face::FSANet *conv_fsanet = new lite::cv::face::FSANet(conv_onnx_path);
  cv::Mat roi = cv::imread(test_img_path);
  lite::cv::types::EulerAngles var_euler_angles, conv_euler_angles;

  // 1. detect euler angles.
  var_fsanet->detect(roi, var_euler_angles);
  conv_fsanet->detect(roi, conv_euler_angles);

  lite::cv::types::EulerAngles euler_angles;

  euler_angles.yaw = (var_euler_angles.yaw + conv_euler_angles.yaw) / 2.0f;
  euler_angles.pitch = (var_euler_angles.pitch + conv_euler_angles.pitch) / 2.0f;
  euler_angles.roll = (var_euler_angles.roll + conv_euler_angles.roll) / 2.0f;
  euler_angles.flag = var_euler_angles.flag && conv_euler_angles.flag;

  if (euler_angles.flag)
  {
    lite::cv::utils::draw_axis_inplace(roi, euler_angles);

    cv::imwrite(save_img_path, roi);

    std::cout << "Default Version"
              << " yaw: " << euler_angles.yaw
              << " pitch: " << euler_angles.pitch
              << " roll: " << euler_angles.roll << std::endl;
  }

  delete var_fsanet;
  delete conv_fsanet;

}

static void test_onnxruntime()
{
  std::string var_onnx_path = "../../../hub/onnx/cv/fsanet-var.onnx";
  std::string conv_onnx_path = "../../../hub/onnx/cv/fsanet-1x1.onnx";
  std::string test_img_path = "../../../examples/lite/resources/test_lite_fsanet.jpg";

  lite::onnxruntime::cv::face::FSANet *var_fsanet = new lite::onnxruntime::cv::face::FSANet(var_onnx_path);
  lite::onnxruntime::cv::face::FSANet *conv_fsanet = new lite::onnxruntime::cv::face::FSANet(conv_onnx_path);
  cv::Mat roi = cv::imread(test_img_path);
  lite::onnxruntime::cv::types::EulerAngles var_euler_angles, conv_euler_angles;

  // 1. detect euler angles.
  var_fsanet->detect(roi, var_euler_angles);
  conv_fsanet->detect(roi, conv_euler_angles);

  lite::onnxruntime::cv::types::EulerAngles euler_angles;

  euler_angles.yaw = (var_euler_angles.yaw + conv_euler_angles.yaw) / 2.0f;
  euler_angles.pitch = (var_euler_angles.pitch + conv_euler_angles.pitch) / 2.0f;
  euler_angles.roll = (var_euler_angles.roll + conv_euler_angles.roll) / 2.0f;
  euler_angles.flag = var_euler_angles.flag && conv_euler_angles.flag;

  if (euler_angles.flag)
  {
    lite::onnxruntime::cv::utils::draw_axis_inplace(roi, euler_angles);

    std::cout << "ONNXRuntime Version"
              << " yaw: " << euler_angles.yaw
              << " pitch: " << euler_angles.pitch
              << " roll: " << euler_angles.roll << std::endl;
  }

  delete var_fsanet;
  delete conv_fsanet;
}

static void test_mnn()
{

}

static void test_ncnn()
{

}

static void test_tnn()
{

}

static void test_lite()
{
  test_default();
  test_onnxruntime();
  test_mnn();
  test_ncnn();
  test_tnn();
}

int main(__unused int argc, __unused char *argv[])
{
  test_lite();
  return 0;
}
