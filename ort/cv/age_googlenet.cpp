//
// Created by DefTruth on 2021/4/2.
//
#include "age_googlenet.h"
#include "ort/core/ort_utils.h"

using ortcv::AgeGoogleNet;

ort::Value AgeGoogleNet::transform(const cv::Mat &mat) {
  cv::Mat canva = mat.clone();
  cv::cvtColor(canva, canva, cv::COLOR_BGR2RGB);
  cv::resize(canva, canva, cv::Size(input_node_dims.at(3),
                                    input_node_dims.at(2)));
  // (1,3,224,224)
  std::cout << "Start normalize.\n";
  ortcv::utils::transform::normalize_inplace(canva, mean_val, scale_val); // float32
  std::cout << "Done normalize.\n";

  return ortcv::utils::transform::mat3f_to_tensor(
      canva, input_node_dims, memory_info_handler,
      input_values_handler, ortcv::utils::transform::CHW);
}

void AgeGoogleNet::detect(const cv::Mat &mat, types::Age &age) {
  if (mat.empty()) return;
  // 1. make input tensor
  ort::Value input_tensor = this->transform(mat);
  // 2. inference scores & boxes.
  std::cout << "Start Detect.\n";
  auto output_tensors = ort_session->Run(
      ort::RunOptions{nullptr}, input_node_names.data(),
      &input_tensor, 1, output_node_names.data(), num_outputs
  );
  std::cout << "Done Detect.\n";
  ort::Value &age_logits = output_tensors.at(0); // (1,8)
  auto age_dims = output_node_dims.at(0);
  float pred_age, pred_prob = -1.f, total_exp = 0.f;
  unsigned int interval = 0;
  const unsigned int num_intervals = age_dims.at(1); // 8
  const float *pred_logits = age_logits.GetTensorMutableData<float>();
  std::vector<float> softmax_probs(num_intervals);
  for (unsigned int i = 0; i < num_intervals; ++i) {
    softmax_probs[i] = std::expf(pred_logits[i]);
    total_exp += softmax_probs[i];
  }
  for (unsigned int i = 0; i < num_intervals; ++i) {
    softmax_probs[i] = softmax_probs[i] / total_exp;
    if (softmax_probs[i] > pred_prob) {
      interval = i;
      pred_prob = softmax_probs[i];
      pred_age = static_cast<float>(
          age_intervals[interval][0] + age_intervals[interval][1]) / 2.0f;
    }
  }
  age.age = pred_age;
  age.age_interval[0] = age_intervals[interval][0];
  age.age_interval[1] = age_intervals[interval][1];
  age.interval_prob = pred_prob;
  age.flag = true;
}