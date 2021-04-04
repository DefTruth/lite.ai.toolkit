//
// Created by DefTruth on 2021/4/4.
//

#include <iostream>
#include <vector>

#include "ort/cv/vgg16_gender.h"
#include "ort/core/ort_utils.h"

static void test_ortcv_vgg16_gender() {
  // download model from: https://drive.google.com/drive/folders/16Z1r7GEXCsJG_384VsjlNxOFXbxcXrqM?usp=sharing
  std::string onnx_path = "../../../hub/onnx/cv/vgg_ilsvrc_16_gender_imdb_wiki.onnx";
  std::string test_img_path = "../../../examples/ort/resources/test_ortcv_vgg16_gender.jpg";
  std::string save_img_path = "../../../logs/test_ortcv_vgg16_gender.jpg";

  ortcv::VGG16Gender *vgg16_gender = new ortcv::VGG16Gender(onnx_path);

  ortcv::types::Gender gender;
  cv::Mat img_bgr = cv::imread(test_img_path);
  vgg16_gender->detect(img_bgr, gender);

  ortcv::utils::draw_gender_inplace(img_bgr, gender);

  cv::imwrite(save_img_path, img_bgr);

  std::cout << "Detected Gender: " << gender.label << std::endl;

  delete vgg16_gender;

}

int main(__unused int argc, __unused char *argv[]) {
  test_ortcv_vgg16_gender();
  return 0;
}