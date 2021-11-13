//
// Created by DefTruth on 2021/4/3.
//

#include "fast_style_transfer.h"
#include "lite/ort/core/ort_utils.h"

using ortcv::FastStyleTransfer;

Ort::Value FastStyleTransfer::transform(const cv::Mat &mat)
{
  cv::Mat canvas;
  cv::resize(mat, canvas, cv::Size(input_node_dims.at(3),
                                   input_node_dims.at(2)));
  cv::cvtColor(canvas, canvas, cv::COLOR_BGR2RGB); // (1,224,224,3)

  return ortcv::utils::transform::create_tensor(
      canvas, input_node_dims, memory_info_handler,
      input_values_handler, ortcv::utils::transform::CHW); // (1,3,224,224)
}

void FastStyleTransfer::detect(const cv::Mat &mat, types::StyleContent &style_content)
{
  // 1. make input tensor
  Ort::Value input_tensor = this->transform(mat);
  // 2. inference
  auto output_tensors = ort_session->Run(
      Ort::RunOptions{nullptr}, input_node_names.data(),
      &input_tensor, 1, output_node_names.data(), num_outputs
  );
  Ort::Value &pred_tensor = output_tensors.at(0); // (1,3,224,224)
  auto pred_dims = output_node_dims.at(0);
  const unsigned int rows = pred_dims.at(2); // H
  const unsigned int cols = pred_dims.at(3); // W

  style_content.mat.create(rows, cols, CV_8UC3); // release & create

  for (unsigned int i = 0; i < rows; ++i)
  {
    cv::Vec3b *p = style_content.mat.ptr<cv::Vec3b>(i);
    for (unsigned int j = 0; j < cols; ++j)
    {
      p[j][0] = cv::saturate_cast<uchar>(pred_tensor.At<float>({0, 0, i, j}));
      p[j][1] = cv::saturate_cast<uchar>(pred_tensor.At<float>({0, 1, i, j}));
      p[j][2] = cv::saturate_cast<uchar>(pred_tensor.At<float>({0, 2, i, j}));
    } // CHW->HWC
  }

  cv::cvtColor(style_content.mat, style_content.mat, cv::COLOR_RGB2BGR); // RGB->BGR

  style_content.flag = true;
}