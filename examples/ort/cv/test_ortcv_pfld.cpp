//
// Created by DefTruth on 2021/3/31.
//

#include <iostream>
#include <vector>

#include "ort/cv/pfld.h"
#include "ort/core/ort_utils.h"


static void test_ortcv_pfld()
{

  std::string onnx_path = "../../../hub/onnx/cv/pfld-106-v3.onnx";
  std::string test_img_path = "../../../examples/ort/resources/test_ortcv_pfld.jpg";
  std::string save_img_path = "../../../logs/test_ortcv_pfld.jpg";

  ortcv::PFLD *pfld = new ortcv::PFLD(onnx_path);

  ortcv::types::Landmarks landmarks;
  cv::Mat img_bgr = cv::imread(test_img_path);
  pfld->detect(img_bgr, landmarks);

  ortcv::utils::draw_landmarks_inplace(img_bgr, landmarks);

  cv::imwrite(save_img_path, img_bgr);

  std::cout << "Detected Landmarks Num: " << landmarks.points.size() << std::endl;

  delete pfld;

}

int main(__unused int argc, __unused char *argv[])
{
  test_ortcv_pfld();
  return 0;
}