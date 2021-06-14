//
// Created by DefTruth on 2021/6/14.
//

#include <iostream>
#include <vector>

#include "ort/cv/fcn_resnet101.h"
#include "ort/core/ort_utils.h"


static void test_ortcv_fcn_resnet101()
{
  std::string onnx_path = "../../../hub/onnx/cv/fcn_resnet101.onnx";
  std::string test_img_path = "../../../examples/ort/resources/test_ortcv_fcn_resnet101.png";
  std::string save_img_path = "../../../logs/test_ortcv_fcn_resnet101.jpg";

  ortcv::FCNResNet101 *fcn_resnet101 = new ortcv::FCNResNet101(onnx_path, 16);

  ortcv::types::SegmentContent content;
  cv::Mat img_bgr = cv::imread(test_img_path);
  fcn_resnet101->detect(img_bgr, content);

  if (content.flag)
  {
    cv::Mat out_img;
    cv::addWeighted(img_bgr, 0.2, content.color_mat,0.8, 0., out_img);
    cv::imwrite(save_img_path, out_img);
    if (!content.names_map.empty())
    {
      for (auto it = content.names_map.begin(); it != content.names_map.end(); ++it)
      {
        std::cout << "Detected Label: " << it->first << " Name: " << it->second << std::endl;
      }
    }
  }

  delete fcn_resnet101;
}

int main(__unused int argc, __unused char *argv[])
{
  test_ortcv_fcn_resnet101();
  return 0;
}