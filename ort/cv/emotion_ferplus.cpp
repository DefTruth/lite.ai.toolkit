//
// Created by DefTruth on 2021/4/3.
//

#include "emotion_ferplus.h"
#include "ort/core/ort_utils.h"

using ortcv::EmotionFerPlus;

ort::Value EmotionFerPlus::transform(const cv::Mat &mat)
{
  cv::Mat canva = mat.clone();
  cv::resize(canva, canva, cv::Size(input_node_dims.at(3),
                                    input_node_dims.at(2)));
  cv::cvtColor(canva, canva, cv::COLOR_BGR2RGB); // (1,3,64,64)
  cv::cvtColor(canva, canva, cv::COLOR_RGB2GRAY); // (1,1,64,64)

  return ortcv::utils::transform::create_tensor(
      canva, input_node_dims, memory_info_handler,
      input_values_handler, ortcv::utils::transform::CHW);
}

void EmotionFerPlus::detect(const cv::Mat &mat, types::Emotions &emotions)
{
  if (mat.empty()) return;
  // 1. make input tensor
  ort::Value input_tensor = this->transform(mat);
  // 2. inference
  auto output_tensors = ort_session->Run(
      ort::RunOptions{nullptr}, input_node_names.data(),
      &input_tensor, 1, output_node_names.data(), num_outputs
  );
  ort::Value &emotion_logits = output_tensors.at(0); // (1,8)
  auto emotion_dims = output_node_dims.at(0);
  unsigned int pred_label = 0;
  const unsigned int num_emotions = emotion_dims.at(1); // 8
  const float *pred_logits = emotion_logits.GetTensorMutableData<float>();
  auto softmax_probs = ortcv::utils::math::softmax<float>(pred_logits, num_emotions, pred_label);
  emotions.label = pred_label;
  emotions.score = softmax_probs[pred_label];
  emotions.text = emotion_texts[pred_label];
  emotions.flag = true;
}