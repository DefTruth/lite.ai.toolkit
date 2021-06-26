//
// Created by DefTruth on 2021/6/26.
//

#include "lite/lite.h"

static void test_default()
{
  std::string onnx_path = "../../../hub/onnx/cv/ssrnet.onnx";
  std::string test_img_path = "../../../examples/lite/resources/test_lite_ssrnet.jpg";
  std::string save_img_path = "../../../logs/test_lite_ssrnet.jpg";

  lite::cv::face::SSRNet *ssrnet = new lite::cv::face::SSRNet(onnx_path);

  lite::cv::types::Age age;
  cv::Mat img_bgr = cv::imread(test_img_path);
  ssrnet->detect(img_bgr, age);

  lite::cv::utils::draw_age_inplace(img_bgr, age);

  cv::imwrite(save_img_path, img_bgr);

  std::cout << "Default Version Done! Detected SSRNet Age: " << age.age << std::endl;

  delete ssrnet;
}

static void test_onnxruntime()
{
  std::string onnx_path = "../../../hub/onnx/cv/ssrnet.onnx";
  std::string test_img_path = "../../../examples/lite/resources/test_lite_ssrnet.jpg";
  std::string save_img_path = "../../../logs/test_onnx_ssrnet.jpg";

  lite::onnxruntime::cv::face::SSRNet *ssrnet = new lite::onnxruntime::cv::face::SSRNet(onnx_path);

  lite::onnxruntime::cv::types::Age age;
  cv::Mat img_bgr = cv::imread(test_img_path);
  ssrnet->detect(img_bgr, age);

  lite::onnxruntime::cv::utils::draw_age_inplace(img_bgr, age);

  cv::imwrite(save_img_path, img_bgr);

  std::cout << "ONNXRuntime Version Done! Detected SSRNet Age: " << age.age << std::endl;

  delete ssrnet;
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
