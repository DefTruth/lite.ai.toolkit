//
// Created by DefTruth on 2021/4/2.
//
#include "age_googlenet.h"
#include "lite/ort/core/ort_utils.h"
#include "lite/utils.h"

using ortcv::AgeGoogleNet;

Ort::Value AgeGoogleNet::transform(const cv::Mat &mat)
{
  cv::Mat canvas;
  cv::resize(mat, canvas, cv::Size(input_node_dims.at(3),
                                   input_node_dims.at(2)));
  cv::cvtColor(canvas, canvas, cv::COLOR_BGR2RGB);
  // (1,3,224,224)
  ortcv::utils::transform::normalize_inplace(canvas, mean_val, scale_val); // float32

  return ortcv::utils::transform::create_tensor(
      canvas, input_node_dims, memory_info_handler,
      input_values_handler, ortcv::utils::transform::CHW);
}

void AgeGoogleNet::detect(const cv::Mat &mat, types::Age &age)
{
  if (mat.empty()) return;
  // 1. make input tensor
  Ort::Value input_tensor = this->transform(mat);
  // 2. inference
  auto output_tensors = ort_session->Run(
      Ort::RunOptions{nullptr}, input_node_names.data(),
      &input_tensor, 1, output_node_names.data(), num_outputs
  );
  Ort::Value &age_logits = output_tensors.at(0); // (1,8)

  auto age_dims = output_node_dims.at(0);
  unsigned int interval = 0;
  const unsigned int num_intervals = age_dims.at(1); // 8

  const float *pred_logits = age_logits.GetTensorMutableData<float>();
  auto softmax_probs = lite::utils::math::softmax<float>(pred_logits, num_intervals, interval);
  const float pred_age = static_cast<float>(age_intervals[interval][0] + age_intervals[interval][1]) / 2.0f;

  age.age = pred_age;
  age.age_interval[0] = age_intervals[interval][0];
  age.age_interval[1] = age_intervals[interval][1];
  age.interval_prob = softmax_probs[interval];
  age.flag = true;
}