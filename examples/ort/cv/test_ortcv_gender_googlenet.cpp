//
// Created by DefTruth on 2021/4/3.
//

#include <iostream>
#include <vector>

#include "ort/cv/gender_googlenet.h"
#include "ort/core/ort_utils.h"

static void test_ortcv_gender_googlenet() {
  std::string onnx_path = "../../../hub/onnx/cv/gender_googlenet.onnx";
  std::string test_img_path = "../../../examples/ort/resources/test_ortcv_gender_googlenet.jpg";
  std::string save_img_path = "../../../logs/test_ortcv_gender_googlenet.jpg";

  ortcv::GenderGoogleNet *gender_googlenet = new ortcv::GenderGoogleNet(onnx_path);

  ortcv::types::Gender gender;
  cv::Mat img_bgr = cv::imread(test_img_path);
  gender_googlenet->detect(img_bgr, gender);

  ortcv::utils::draw_gender_inplace(img_bgr, gender);

  cv::imwrite(save_img_path, img_bgr);

  std::cout << "Detected Gender: " << gender.label << std::endl;

  delete gender_googlenet;

}

int main(__unused int argc, __unused char *argv[]) {
  test_ortcv_gender_googlenet();
  return 0;
}