//
// Created by DefTruth on 2021/4/4.
//

#include "vgg16_gender.h"
#include "ort/core/ort_utils.h"

using ortcv::VGG16Gender;

ort::Value VGG16Gender::transform(const cv::Mat &mat)
{
  cv::Mat canva = mat.clone();
  cv::resize(canva, canva, cv::Size(input_node_dims.at(3),
                                    input_node_dims.at(2)));
  cv::cvtColor(canva, canva, cv::COLOR_BGR2RGB);   // (1,3,224,224)

  return ortcv::utils::transform::mat3f_to_tensor(
      canva, input_node_dims, memory_info_handler,
      input_values_handler, ortcv::utils::transform::CHW);
}

void VGG16Gender::detect(const cv::Mat &mat, types::Gender &gender)
{
  if (mat.empty()) return;
  // 1. make input tensor
  ort::Value input_tensor = this->transform(mat);
  // 2. inference.
  auto output_tensors = ort_session->Run(
      ort::RunOptions{nullptr}, input_node_names.data(),
      &input_tensor, 1, output_node_names.data(), num_outputs
  );
  ort::Value &gender_logits = output_tensors.at(0); // (1,2)
  auto gender_dims = output_node_dims.at(0);
  const unsigned int num_genders = gender_dims.at(1); // 2
  unsigned int pred_gender = 0;
  const float *pred_logits = gender_logits.GetTensorMutableData<float>();
  auto softmax_probs = ortcv::utils::math::softmax<float>(pred_logits, num_genders, pred_gender);
  gender.label = pred_gender;
  gender.text = gender_texts[pred_gender];
  gender.score = softmax_probs[pred_gender];
  gender.flag = true;
}