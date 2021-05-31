//
// Created by DefTruth on 2021/3/14.
//

#include "fsanet.h"
#include "ort/core/ort_utils.h"

using ortcv::FSANet;

ort::Value FSANet::transform(const cv::Mat &mat)
{
  cv::Mat canva;
  // 0. padding
  const int h = mat.rows;
  const int w = mat.cols;
  const int nh = static_cast<int>((static_cast<float>(h) + pad * static_cast<float>(h)));
  const int nw = static_cast<int>((static_cast<float>(w) + pad * static_cast<float>(w)));

  const int nx1 = std::max(0, static_cast<int>((nw - w) / 2));
  const int ny1 = std::max(0, static_cast<int>((nh - h) / 2));

  canva = cv::Mat(nh, nw, CV_8UC3, cv::Scalar(0, 0, 0));
  mat.copyTo(canva(cv::Rect(nx1, ny1, w, h)));

  cv::resize(canva, canva, cv::Size(input_width, input_height));
  ortcv::utils::transform::normalize_inplace(canva, 127.5, 1.f / 127.5f);

  return ortcv::utils::transform::create_tensor(
      canva, input_node_dims, memory_info_handler,
      input_values_handler, ortcv::utils::transform::CHW);
}

void FSANet::detect(const cv::Mat &mat, types::EulerAngles &euler_angles)
{

  ort::Value input_tensor = this->transform(mat);

  auto output_tensors = ort_session->Run(
      ort::RunOptions{nullptr}, input_node_names.data(),
      &input_tensor, 1, output_node_names.data(), 1
  );

  const float *angles = output_tensors.front().GetTensorMutableData<float>();

  euler_angles.yaw = angles[0];
  euler_angles.pitch = angles[1];
  euler_angles.roll = angles[2];
  euler_angles.flag = true;
}


















































