//
// Created by DefTruth on 2021/4/7.
//

#include <iostream>
#include <vector>

#include "ort/cv/ssrnet.h"
#include "ort/core/ort_utils.h"

static void test_ortcv_ssrnet()
{
  std::string onnx_path = "../../../hub/onnx/cv/ssrnet.onnx";
  std::string test_img_path = "../../../examples/ort/resources/test_ortcv_ssrnet.jpg";
  std::string save_img_path = "../../../logs/test_ortcv_ssrnet.jpg";

  ortcv::SSRNet *ssrnet = new ortcv::SSRNet(onnx_path);

  ortcv::types::Age age;
  cv::Mat img_bgr = cv::imread(test_img_path);
  ssrnet->detect(img_bgr, age);

  ortcv::utils::draw_age_inplace(img_bgr, age);

  cv::imwrite(save_img_path, img_bgr);

  std::cout << "Detected SSRNet Age: " << age.age << std::endl;

  delete ssrnet;

}

int main(__unused int argc, __unused char *argv[])
{
  test_ortcv_ssrnet();
  return 0;
}