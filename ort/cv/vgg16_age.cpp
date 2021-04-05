//
// Created by DefTruth on 2021/4/4.
//

#include "vgg16_age.h"
#include "ort/core/ort_utils.h"

using ortcv::VGG16Age;

ort::Value VGG16Age::transform(const cv::Mat &mat)
{
  cv::Mat canva = mat.clone();
  cv::resize(canva, canva, cv::Size(input_node_dims.at(3),
                                    input_node_dims.at(2)));
  cv::cvtColor(canva, canva, cv::COLOR_BGR2RGB);   // (1,3,224,224)

  return ortcv::utils::transform::create_tensor(
      canva, input_node_dims, memory_info_handler,
      input_values_handler, ortcv::utils::transform::CHW);
}

void VGG16Age::detect(const cv::Mat &mat, types::Age &age)
{
  // 1. make input tensor
  ort::Value input_tensor = this->transform(mat);
  // 2. inference
  auto output_tensors = ort_session->Run(
      ort::RunOptions{nullptr}, input_node_names.data(),
      &input_tensor, 1, output_node_names.data(), num_outputs
  );
  float pred_age = 0.f, top10_pred_prob = 0.f;
  ort::Value &age_probs = output_tensors.at(0); // (1,101)
  auto age_dims = output_node_dims.at(0); // (1,101)
  const unsigned int num_intervals = age_dims.at(1); // 101
  std::vector<float> pred_probs(num_intervals);
  for (unsigned int i = 0; i < num_intervals; ++i)
  {
    float cur_prob = age_probs.At<float>({0, i});
    pred_age += cur_prob * static_cast<float>(i);
    pred_probs[i] = cur_prob;
  }

  std::sort(pred_probs.begin(), pred_probs.end(), std::greater<float>());
  for (unsigned int i = 0; i < 10; ++i) top10_pred_prob += pred_probs[i];

  const unsigned int interval_min = static_cast<int>(pred_age - 2.f > 0.f ? pred_age - 2.f : 0.f);
  const unsigned int interval_max = static_cast<int>(pred_age + 3.f < 100.f ? pred_age + 3.f : 100.f);

  age.age = pred_age;
  age.age_interval[0] = interval_min;
  age.age_interval[1] = interval_max;
  age.interval_prob = top10_pred_prob;
  age.flag = true;
}