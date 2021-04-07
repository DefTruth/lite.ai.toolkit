//
// Created by DefTruth on 2021/3/14.
//

#include "ssrnet.h"
#include "ort/core/ort_utils.h"

using ortcv::SSRNet;

ort::Value SSRNet::transform(const cv::Mat &mat)
{
  cv::Mat canva = mat.clone();
  cv::resize(canva, canva, cv::Size(input_node_dims.at(3),
                                    input_node_dims.at(2))); // 64x64x3
  canva.convertTo(canva, CV_32FC3, 1.0f / 255.0f, 0.f);  // 64x64x3 (0.,1.0)
  // (1,3,64,64)
  ortcv::utils::transform::normalize_inplace(canva, mean_val, scale_val); // float32

  return ortcv::utils::transform::create_tensor(
      canva, input_node_dims, memory_info_handler,
      input_values_handler, ortcv::utils::transform::CHW);
}

void SSRNet::detect(const cv::Mat &mat, types::Age &age)
{
  if (mat.empty()) return;
  // 1. make input tensor
  ort::Value input_tensor = this->transform(mat);
  // 2. inference
  auto output_tensors = ort_session->Run(
      ort::RunOptions{nullptr}, input_node_names.data(),
      &input_tensor, 1, output_node_names.data(), num_outputs
  );
  ort::Value &age_tensor = output_tensors.at(0); // (1,)
  const float pred_age = age_tensor.At<float>({0});
  const unsigned int interval_min = static_cast<int>(pred_age - 2.f > 0.f ? pred_age - 2.f : 0.f);
  const unsigned int interval_max = static_cast<int>(pred_age + 3.f < 100.f ? pred_age + 3.f : 100.f);

  age.age = pred_age;
  age.age_interval[0] = interval_min;
  age.age_interval[1] = interval_max;
  age.interval_prob = 1.0f;
  age.flag = true;
}