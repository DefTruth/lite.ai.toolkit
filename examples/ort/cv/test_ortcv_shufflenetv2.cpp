//
// Created by DefTruth on 2021/5/30.
//

#include <iostream>
#include <vector>

#include "ort/cv/shufflenetv2.h"
#include "ort/core/ort_utils.h"

static void test_ortcv_shufflenetv2()
{
  std::string onnx_path = "../../../hub/onnx/cv/shufflenet-v2-10.onnx";
  std::string test_img_path = "../../../examples/ort/resources/test_ortcv_shufflenetv2.jpg";

  ortcv::ShuffleNetV2 *shufflenetv2 = new ortcv::ShuffleNetV2(onnx_path);

  ortcv::types::ImageNetContent content;
  cv::Mat img_bgr = cv::imread(test_img_path);
  shufflenetv2->detect(img_bgr, content);

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

  delete shufflenetv2;
}

int main(__unused int argc, __unused char *argv[])
{
  test_ortcv_shufflenetv2();
  return 0;
}