//
// Created by DefTruth on 2021/4/9.
//

#include "colorizer.h"
#include "ort/core/ort_utils.h"

using ortcv::Colorizer;


ort::Value Colorizer::transform(const cv::Mat &mat)
{
  cv::Mat mat_l; // assume that input mat is L of Lab
  mat.convertTo(mat_l, CV_32FC1, 1.0f, 0.f); // (256,256,1) range (0.,100.)

  return ortcv::utils::transform::create_tensor(
      mat_l, input_node_dims, memory_info_handler,
      input_values_handler, ortcv::utils::transform::CHW
  ); // (1,1,256,256)
}

void Colorizer::detect(const cv::Mat &mat, types::ColorizeContent &colorize_content)
{
  if (mat.empty()) return;
  const unsigned int height = mat.rows;
  const unsigned int width = mat.cols;

  cv::Mat mat_rs = mat.clone();
  cv::resize(mat_rs, mat_rs, cv::Size(input_node_dims.at(3), input_node_dims.at(2))); // (256,256,3)
  cv::Mat mat_rs_norm, mat_orig_norm;
  mat_rs.convertTo(mat_rs_norm, CV_32FC3, 1.0f / 255.0f, 0.f); // (0.,1.) BGR
  mat.convertTo(mat_orig_norm, CV_32FC3, 1.0f / 255.0f, 0.f); // (0.,1.) BGR
  if (mat_rs_norm.empty() || mat_orig_norm.empty()) return;

  cv::Mat mat_lab_orig, mat_lab_rs;
  cv::cvtColor(mat_rs_norm, mat_lab_rs, cv::COLOR_BGR2Lab);
  cv::cvtColor(mat_orig_norm, mat_lab_orig, cv::COLOR_BGR2Lab);

  cv::Mat mat_rs_l, mat_orig_l;
  std::vector<cv::Mat> mats_rs_lab, mats_orig_lab;
  cv::split(mat_lab_rs, mats_rs_lab);
  cv::split(mat_lab_orig, mats_orig_lab);

  mat_rs_l = mats_rs_lab.at(0);
  mat_orig_l = mats_orig_lab.at(0);

  // 1. make input tensor
  ort::Value input_tensor = this->transform(mat_rs_l); // (1,1,256,256)
  auto output_tensors = ort_session->Run(
      ort::RunOptions{nullptr}, input_node_names.data(),
      &input_tensor, 1, output_node_names.data(), num_outputs
  );
  ort::Value &pred_ab_tensor = output_tensors.at(0); // (1,2,256,256)
  auto pred_dims = output_node_dims.at(0);
  const unsigned int rows = pred_dims.at(2); // H 256
  const unsigned int cols = pred_dims.at(3); // W 256

  cv::Mat out_a_orig(rows, cols, CV_32FC1);
  cv::Mat out_b_orig(rows, cols, CV_32FC1);

  for (unsigned int i = 0; i < rows; ++i)
  {
    float *pa = out_a_orig.ptr<float>(i);
    float *pb = out_b_orig.ptr<float>(i);
    for (unsigned int j = 0; j < cols; ++j)
    {
      pa[j] = pred_ab_tensor.At<float>({0, 0, i, j});
      pb[j] = pred_ab_tensor.At<float>({0, 1, i, j});
    } // CHW->HWC
  }

  if (rows != height || cols != width)
  {
    cv::resize(out_a_orig, out_a_orig, cv::Size(width, height));
    cv::resize(out_b_orig, out_b_orig, cv::Size(width, height));
  }

  std::vector<cv::Mat> out_mats_lab;
  out_mats_lab.push_back(mat_orig_l);
  out_mats_lab.push_back(out_a_orig);
  out_mats_lab.push_back(out_b_orig);

  cv::Mat merge_mat_lab, mat_bgr_norm;
  cv::merge(out_mats_lab, merge_mat_lab);
  if (merge_mat_lab.empty()) return;
  cv::cvtColor(merge_mat_lab, mat_bgr_norm, cv::COLOR_Lab2BGR); // CV_32FC3
  mat_bgr_norm *= 255.0f;

  mat_bgr_norm.convertTo(colorize_content.mat, CV_8UC3); // uint8

  colorize_content.flag = true;

}