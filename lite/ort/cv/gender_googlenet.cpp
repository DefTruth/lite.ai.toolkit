//
// Created by DefTruth on 2021/4/3.
//
#include "gender_googlenet.h"
#include "lite/ort/core/ort_utils.h"
#include "lite/utils.h"

using ortcv::GenderGoogleNet;

Ort::Value GenderGoogleNet::transform(const cv::Mat &mat)
{
  cv::Mat canvas;
  cv::cvtColor(mat, canvas, cv::COLOR_BGR2RGB);
  cv::resize(canvas, canvas, cv::Size(input_node_dims.at(3),
                                      input_node_dims.at(2)));
  // (1,3,224,224)
  ortcv::utils::transform::normalize_inplace(canvas, mean_val, scale_val); // float32

  return ortcv::utils::transform::create_tensor(
      canvas, input_node_dims, memory_info_handler,
      input_values_handler, ortcv::utils::transform::CHW);
}

void GenderGoogleNet::detect(const cv::Mat &mat, types::Gender &gender)
{
  if (mat.empty()) return;
  // 1. make input tensor
  Ort::Value input_tensor = this->transform(mat);
  // 2. inference.
  auto output_tensors = ort_session->Run(
      Ort::RunOptions{nullptr}, input_node_names.data(),
      &input_tensor, 1, output_node_names.data(), num_outputs
  );
  Ort::Value &gender_logits = output_tensors.at(0); // (1,2)
  auto gender_dims = output_node_dims.at(0);
  const unsigned int num_genders = gender_dims.at(1); // 2
  unsigned int pred_gender = 0;
  const float *pred_logits = gender_logits.GetTensorMutableData<float>();
  auto softmax_probs = lite::utils::math::softmax<float>(pred_logits, num_genders, pred_gender);
  unsigned int gender_label = pred_gender == 1 ? 0 : 1;
  gender.label = gender_label;
  gender.text = gender_texts[gender_label];
  gender.score = softmax_probs[pred_gender];
  gender.flag = true;
}
