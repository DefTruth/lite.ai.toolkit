//
// Created by DefTruth on 2021/4/4.
//

#include <iostream>
#include <vector>

#include "ort/cv/vgg16_age.h"
#include "ort/core/ort_utils.h"

static void test_ortcv_vgg16_age()
{
  std::string onnx_path = "../../../hub/onnx/cv/vgg_ilsvrc_16_age_imdb_wiki.onnx";
  std::string test_img_path = "../../../examples/ort/resources/test_ortcv_vgg16_age.jpg";
  std::string save_img_path = "../../../logs/test_ortcv_vgg16_age.jpg";

  ortcv::VGG16Age *vgg16_age = new ortcv::VGG16Age(onnx_path);

  ortcv::types::Age age;
  cv::Mat img_bgr = cv::imread(test_img_path);
  vgg16_age->detect(img_bgr, age);

  ortcv::utils::draw_age_inplace(img_bgr, age);

  cv::imwrite(save_img_path, img_bgr);

  std::cout << "Detected Age: " << age.age << std::endl;

  delete vgg16_age;

}

int main(__unused int argc, __unused char *argv[])
{
  test_ortcv_vgg16_age();
  return 0;
}