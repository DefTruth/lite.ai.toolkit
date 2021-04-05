//
// Created by DefTruth on 2021/4/3.
//

#include "fast_style_transfer.h"
#include "ort/core/ort_utils.h"

using ortcv::FastStyleTransfer;

ort::Value FastStyleTransfer::transform(const cv::Mat &mat)
{
  cv::Mat canva = mat.clone();
  cv::resize(canva, canva, cv::Size(input_node_dims.at(3),
                                    input_node_dims.at(2)));
  cv::cvtColor(canva, canva, cv::COLOR_BGR2RGB); // (1,224,224,3)

  return ortcv::utils::transform::mat3f_to_tensor(
      canva, input_node_dims, memory_info_handler,
      input_values_handler, ortcv::utils::transform::CHW); // (1,3,224,224)
}

void FastStyleTransfer::delect(const cv::Mat &mat, types::StyleContent &style_content)
{
  // 1. make input tensor
  ort::Value input_tensor = this->transform(mat);
  // 2. inference
  auto output_tensors = ort_session->Run(
      ort::RunOptions{nullptr}, input_node_names.data(),
      &input_tensor, 1, output_node_names.data(), num_outputs
  );
  ort::Value &pred_tensor = output_tensors.at(0); // (1,3,224,224)
  auto pred_dims = output_node_dims.at(0);
  const unsigned int channels = pred_dims.at(1); // C
  const unsigned int rows = pred_dims.at(2); // H
  const unsigned int cols = pred_dims.at(3); // W

  style_content.style_mat.create(rows, cols, CV_8UC(channels)); // release & create

  if (style_content.style_mat.isContinuous())
  {
    for (unsigned int k = 0; k < channels; ++k)
      for (unsigned int i = 0; i < rows; ++i)
        for (unsigned int j = 0; j < cols; ++j)
          style_content.style_mat.data[i * cols + j * channels + k] =
              cv::saturate_cast<uchar>(pred_tensor.At<float>({0, k, i, j}));
  } else
  {
    for (unsigned int k = 0; k < channels; ++k)
      for (unsigned int i = 0; i < rows; ++i)
        for (unsigned int j = 0; j < cols; ++j)
          style_content.style_mat.at<uchar>(i, j, k) =
              cv::saturate_cast<uchar>(pred_tensor.At<float>({0, k, i, j}));
  }

  cv::cvtColor(style_content.style_mat, style_content.style_mat, cv::COLOR_RGB2BGR); // RGB->BGR

  style_content.flag = true;
}