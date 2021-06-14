//
// Created by DefTruth on 2021/6/14.
//

#include <iostream>
#include <vector>

#include "ort/cv/ibnnet.h"
#include "ort/core/ort_utils.h"

static void test_ortcv_ibnnet()
{
  std::string onnx_path = "../../../hub/onnx/cv/ibnnet18.onnx";
  std::string test_img_path = "../../../examples/ort/resources/test_ortcv_ibnnet.jpg";

  ortcv::IBNNet *ibnnet = new ortcv::IBNNet(onnx_path);

  ortcv::types::ImageNetContent content;
  cv::Mat img_bgr = cv::imread(test_img_path);
  ibnnet->detect(img_bgr, content);

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

  delete ibnnet;
}

int main(__unused int argc, __unused char *argv[])
{
  test_ortcv_ibnnet();
  return 0;
}