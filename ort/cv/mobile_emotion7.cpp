//
// Created by DefTruth on 2021/7/24.
//

#include "mobile_emotion7.h"
#include "ort/core/ort_utils.h"

using ortcv::MobileEmotion7;

Ort::Value MobileEmotion7::transform(const cv::Mat &mat)
{
  cv::Mat canva = mat.clone();
  cv::resize(canva, canva, cv::Size(input_node_dims.at(3),
                                    input_node_dims.at(2)));
  cv::cvtColor(canva, canva, cv::COLOR_BGR2RGB);
  // (1,3,224,224)
  ortcv::utils::transform::normalize_inplace(canva, mean_vals, scale_vals);
  return ortcv::utils::transform::create_tensor(
      canva, input_node_dims, memory_info_handler,
      input_values_handler, ortcv::utils::transform::HWC);
}

void MobileEmotion7::detect(const cv::Mat &mat, types::Emotions &emotions)
{
  if (mat.empty()) return;
  // 1. make input tensor
  Ort::Value input_tensor = this->transform(mat);
  // 2. inference
  auto output_tensors = ort_session->Run(
      Ort::RunOptions{nullptr}, input_node_names.data(),
      &input_tensor, 1, output_node_names.data(), num_outputs
  );
  Ort::Value &emotion_logits = output_tensors.at(0); // (1,7)
  auto emotion_dims = output_node_dims.at(0);
  const float *pred_probs = emotion_logits.GetTensorMutableData<float>(); // output with softmax
  const unsigned int num_emotions = emotion_dims.at(1); // 7
  unsigned int pred_label = 0;
  float pred_score = pred_probs[0];

  for (unsigned int i = 0; i < num_emotions; ++i)
  {
    if (pred_probs[i] > pred_score)
    {
      pred_score = pred_probs[i];
      pred_label = i;
    }
  }

  emotions.label = pred_label;
  emotions.score = pred_score;
  emotions.text = emotion_texts[pred_label];
  emotions.flag = true;
}