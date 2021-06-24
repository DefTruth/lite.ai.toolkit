//
// Created by DefTruth on 2021/6/24.
//

#include "lite/lite.h"

static void test_lite_age_googlenet()
{
  std::string onnx_path = "../../../hub/onnx/cv/age_googlenet.onnx";
  std::string test_img_path = "../../../examples/lite/resources/test_lite_age_googlenet.jpg";
  std::string save_img_path = "../../../logs/test_lite_age_googlenet.jpg";

  // 1. Test Default Engine ONNXRuntime
  lite::cv::face::AgeGoogleNet *age_googlenet = new lite::cv::face::AgeGoogleNet(onnx_path);

  lite::cv::types::Age age;
  cv::Mat img_bgr = cv::imread(test_img_path);
  age_googlenet->detect(img_bgr, age);

  lite::cv::utils::draw_age_inplace(img_bgr, age);

  cv::imwrite(save_img_path, img_bgr);

  std::cout << "Default Version Detected Age: " << age.age << std::endl;

  delete age_googlenet;

  // 2. Test Specific Engine ONNXRuntime
  lite::onnxruntime::cv::face::AgeGoogleNet *onnx_age_googlenet =
      new lite::onnxruntime::cv::face::AgeGoogleNet(onnx_path);
  lite::onnxruntime::cv::types::Age age_;
  onnx_age_googlenet->detect(img_bgr, age_);

  std::cout << "ONNXRuntime Version Detected Age: " << age_.age << std::endl;

  delete onnx_age_googlenet;

}

int main(__unused int argc, __unused char *argv[])
{
  test_lite_age_googlenet();
  return 0;
}
