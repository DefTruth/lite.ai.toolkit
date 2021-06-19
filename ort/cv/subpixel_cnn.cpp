//
// Created by DefTruth on 2021/4/5.
//

#include "subpixel_cnn.h"
#include "ort/core/ort_utils.h"

using ortcv::SubPixelCNN;

Ort::Value SubPixelCNN::transform(const cv::Mat &mat)
{
  cv::Mat mat_y; // assume that input mat is Y of YCrCb
  mat.convertTo(mat_y, CV_32FC1, 1.0f / 255.0f, 0.f); // (224,224,1) range (0.,1.0)

  return ortcv::utils::transform::create_tensor(
      mat_y, input_node_dims, memory_info_handler,
      input_values_handler, ortcv::utils::transform::CHW
  );
}


void SubPixelCNN::detect(const cv::Mat &mat, types::SuperResolutionContent &super_resolution_content)
{
  if (mat.empty()) return;
  cv::Mat mat_copy = mat.clone();
  cv::resize(mat_copy, mat_copy, cv::Size(input_node_dims.at(3), input_node_dims.at(2))); // (224,224,3)
  cv::Mat mat_ycrcb, mat_y, mat_cr, mat_cb;
  cv::cvtColor(mat_copy, mat_ycrcb, cv::COLOR_BGR2YCrCb);

  // 0. split
  std::vector<cv::Mat> split_mats;
  cv::split(mat_ycrcb, split_mats);
  mat_y = split_mats.at(0); // (224,224,1) uchar CV_8UC1
  mat_cr = split_mats.at(1);
  mat_cb = split_mats.at(2);

  // 1. make input tensor
  Ort::Value input_tensor = this->transform(mat_y);
  // 2. inference
  auto output_tensors = ort_session->Run(
      Ort::RunOptions{nullptr}, input_node_names.data(),
      &input_tensor, 1, output_node_names.data(), num_outputs
  );
  Ort::Value &pred_tensor = output_tensors.at(0); // (1,1,672,672)
  auto pred_dims = output_node_dims.at(0);
  const unsigned int rows = pred_dims.at(2); // H
  const unsigned int cols = pred_dims.at(3); // W

  mat_y.create(rows, cols, CV_8UC1); // release & create

  for (unsigned int i = 0; i < rows; ++i)
  {
    uchar *p = mat_y.ptr<uchar>(i);
    for (unsigned int j = 0; j < cols; ++j)
    {
      p[j] = cv::saturate_cast<uchar>(pred_tensor.At<float>({0, 0, i, j}) * 255.0f);

    } // CHW->HWC
  }

  cv::resize(mat_cr, mat_cr, cv::Size(cols, rows));
  cv::resize(mat_cb, mat_cb, cv::Size(cols, rows));

  std::vector<cv::Mat> out_mats;
  out_mats.push_back(mat_y);
  out_mats.push_back(mat_cr);
  out_mats.push_back(mat_cb);

  // 3. merge
  cv::merge(out_mats, super_resolution_content.mat);
  if (super_resolution_content.mat.empty())
  {
    super_resolution_content.flag = false;
    return;
  }
  cv::cvtColor(super_resolution_content.mat, super_resolution_content.mat, cv::COLOR_YCrCb2BGR);
  super_resolution_content.flag = true;
}