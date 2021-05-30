//
// Created by DefTruth on 2021/5/30.
//

#include "efficientnet_lite4.h"
#include "ort/core/ort_utils.h"

using ortcv::EfficientNetLite4;


ort::Value EfficientNetLite4::transform(const cv::Mat &mat)
{
  cv::Mat canva = mat.clone();
  cv::resize(canva, canva, cv::Size(input_node_dims.at(2),
                                    input_node_dims.at(1))); // (1,h,w,c)
  cv::cvtColor(canva, canva, cv::COLOR_BGR2RGB);
  ortcv::utils::transform::normalize_inplace(canva, mean_val, scale_val); // float32
  // (1,224,224,3)
  return ortcv::utils::transform::create_tensor(
      canva, input_node_dims, memory_info_handler,
      input_values_handler, ortcv::utils::transform::HWC);
}

void EfficientNetLite4::detect(const cv::Mat &mat, types::ImageNetContent &content, unsigned int top_k)
{
  if (mat.empty()) return;
  // 1. make input tensor
  ort::Value input_tensor = this->transform(mat);
  // 2. inference
  auto output_tensors = ort_session->Run(
      ort::RunOptions{nullptr}, input_node_names.data(),
      &input_tensor, 1, output_node_names.data(), num_outputs
  );
  ort::Value &logits_tensor = output_tensors.at(0); // (1,1000)
  const unsigned int num_classes = output_node_dims.at(0).at(1);
  const float *scores = logits_tensor.GetTensorMutableData<float>(); // float
  std::vector<unsigned int> sorted_indices = ortcv::utils::math::argsort<float>(scores, num_classes);
  if (top_k > num_classes) top_k = num_classes;

  content.scores.clear();
  content.labels.clear();
  content.texts.clear();
  for (unsigned int i = 0; i < top_k; ++i)
  {
    content.labels.push_back(sorted_indices[i]);
    content.scores.push_back(scores[sorted_indices[i]]);
    content.texts.push_back(class_names[sorted_indices[i]]);
  }
  content.flag = true;
}