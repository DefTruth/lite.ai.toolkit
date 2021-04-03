//
// Created by DefTruth on 2021/4/3.
//
#include "gender_googlenet.h"
#include "ort/core/ort_utils.h"

using ortcv::GenderGoogleNet;

ort::Value GenderGoogleNet::transform(const cv::Mat &mat) {
  cv::Mat canva = mat.clone();
  cv::cvtColor(canva, canva, cv::COLOR_BGR2RGB);
  cv::resize(canva, canva, cv::Size(input_node_dims.at(3),
                                    input_node_dims.at(2)));
  // (1,3,224,224)
  ortcv::utils::transform::normalize_inplace(canva, mean_val, scale_val); // float32

  return ortcv::utils::transform::mat3f_to_tensor(
      canva, input_node_dims, memory_info_handler,
      input_values_handler, ortcv::utils::transform::CHW);
}

void GenderGoogleNet::detect(const cv::Mat &mat, types::Gender &gender) {
  if (mat.empty()) return;
  // 1. make input tensor
  ort::Value input_tensor = this->transform(mat);
  // 2. inference scores & boxes.
  auto output_tensors = ort_session->Run(
      ort::RunOptions{nullptr}, input_node_names.data(),
      &input_tensor, 1, output_node_names.data(), num_outputs
  );
  ort::Value &gender_logits = output_tensors.at(0); // (1,2)
  auto gender_dims = output_node_dims.at(0);
  const unsigned int num_genders = gender_dims.at(1); // 2
  float pred_prob = -1.f, total_exp = 0.f;
  unsigned int pred_gender = 0;
  const float *pred_logits = gender_logits.GetTensorMutableData<float>();
  std::vector<float> softmax_probs(num_genders);
  for (unsigned int i = 0; i < num_genders; ++i) {
    softmax_probs[i] = std::expf(pred_logits[i]);
    total_exp += softmax_probs[i];
  }
  for (unsigned int i = 0; i < num_genders; ++i) {
    softmax_probs[i] = softmax_probs[i] / total_exp;
    if (softmax_probs[i] > pred_prob) {
      pred_gender = i;
      pred_prob = softmax_probs[i];
    }
  }
  unsigned gender_label = pred_gender == 1 ? 0 : 1;
  gender.label = gender_label;
  gender.text = gender_texts[gender_label];
  gender.score = pred_prob;
  gender.flag = true;
}
