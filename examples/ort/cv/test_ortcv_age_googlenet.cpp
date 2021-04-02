//
// Created by DefTruth on 2021/4/2.
//

#include <iostream>
#include <vector>

#include "ort/cv/age_googlenet.h"
#include "ort/core/ort_utils.h"

static void test_ortcv_age_googlenet() {
  std::string onnx_path = "../../../hub/onnx/cv/age_googlenet.onnx";
  std::string test_img_path = "../../../examples/ort/resources/test_ortcv_age_googlenet.jpg";
  std::string save_img_path = "../../../logs/test_ortcv_age_googlenet.jpg";

  ortcv::AgeGoogleNet *age_googlenet = new ortcv::AgeGoogleNet(onnx_path);

  ortcv::types::Age age;
  cv::Mat img_bgr = cv::imread(test_img_path);
  age_googlenet->detect(img_bgr, age);

  ortcv::utils::draw_age_inplace(img_bgr, age);

  cv::imwrite(save_img_path, img_bgr);

  std::cout << "Detected Age: " << age.age << std::endl;

  delete age_googlenet;

}

int main(__unused int argc, __unused char *argv[]) {
  test_ortcv_age_googlenet();
  return 0;
}