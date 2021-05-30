//
// Created by DefTruth on 2021/5/30.
//

#include <iostream>
#include <vector>

#include "ort/cv/efficientnet_lite4.h"
#include "ort/core/ort_utils.h"

static void test_ortcv_efficientnet_lite4()
{
  std::string onnx_path = "../../../hub/onnx/cv/efficientnet-lite4-11.onnx";
  std::string test_img_path = "../../../examples/ort/resources/test_ortcv_efficientnet_lite4.jpg";

  ortcv::EfficientNetLite4 *efficientnet_lite4 = new ortcv::EfficientNetLite4(onnx_path);

  ortcv::types::ImageNetContent content;
  cv::Mat img_bgr = cv::imread(test_img_path);
  efficientnet_lite4->detect(img_bgr, content);

  if (content.flag)
  {
    const unsigned int top_k = content.scores.size();
    if (top_k > 0)
    {
      for (unsigned int i = 0; i < top_k; ++i)
        std::cout << i + 1
                  << ": " << content.labels.at(i)
                  << ": " << content.texts.at(i)
                  << ": " << content.scores.at(i)
                  << std::endl;
    }
  }

  delete efficientnet_lite4;
}

int main(__unused int argc, __unused char *argv[])
{
  test_ortcv_efficientnet_lite4();
  return 0;
}