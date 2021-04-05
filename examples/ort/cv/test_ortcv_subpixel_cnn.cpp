//
// Created by DefTruth on 2021/4/5.
//

#include <iostream>
#include <vector>

#include "ort/cv/subpixel_cnn.h"
#include "ort/core/ort_utils.h"

static void test_ortcv_subpixel_cnn()
{
  std::string onnx_path = "../../../hub/onnx/cv/subpixel-cnn.onnx";
  std::string test_img_path = "../../../examples/ort/resources/test_ortcv_subpixel_cnn.jpg";
  std::string save_img_path = "../../../logs/test_ortcv_subpixel_cnn.jpg";

  ortcv::SubPixelCNN *subpixel_cnn = new ortcv::SubPixelCNN(onnx_path);

  ortcv::types::SuperResolutionContent super_resolution_content;
  cv::Mat img_bgr = cv::imread(test_img_path);
  subpixel_cnn->detect(img_bgr, super_resolution_content);

  if (super_resolution_content.flag) cv::imwrite(save_img_path, super_resolution_content.mat);

  std::cout << "Super Resolution Done." << std::endl;

  delete subpixel_cnn;
}

int main(__unused int argc, __unused char *argv[])
{
  test_ortcv_subpixel_cnn();
  return 0;
}