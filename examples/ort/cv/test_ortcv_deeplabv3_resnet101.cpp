//
// Created by DefTruth on 2021/6/11.
//

#include <iostream>
#include <vector>

#include "ort/cv/deeplabv3_resnet101.h"
#include "ort/core/ort_utils.h"


static void test_ortcv_deeplabv3_resnet101()
{
  std::string onnx_path = "../../../hub/onnx/cv/deeplabv3_resnet101_coco.onnx";
  std::string test_img_path = "../../../examples/ort/resources/test_ortcv_deeplabv3_resnet101.png";
  std::string save_img_path = "../../../logs/test_ortcv_deeplabv3_resnet101.jpg";

  ortcv::DeepLabV3ResNet101 *deeplabv3_resnet101 = new ortcv::DeepLabV3ResNet101(onnx_path, 16);

  ortcv::types::SegmentContent content;
  cv::Mat img_bgr = cv::imread(test_img_path);
  deeplabv3_resnet101->detect(img_bgr, content);

  if (content.flag)
  {
    cv::imwrite(save_img_path, content.color_mat);
    if (!content.names_map.empty())
    {
      for (auto it = content.names_map.begin(); it != content.names_map.end(); ++it)
      {
        std::cout << "Detected Label: " << it->first << " Name: " << it->second << std::endl;
      }
    }
  }

  delete deeplabv3_resnet101;
}

int main(__unused int argc, __unused char *argv[])
{
  test_ortcv_deeplabv3_resnet101();
  return 0;
}