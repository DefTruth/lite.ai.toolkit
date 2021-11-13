//
// Created by DefTruth on 2021/3/14.
//

#include "ssrnet.h"
#include "lite/ort/core/ort_utils.h"

using ortcv::SSRNet;

Ort::Value SSRNet::transform(const cv::Mat &mat)
{
  cv::Mat canvas;
  cv::resize(mat, canvas, cv::Size(input_node_dims.at(3),
                                   input_node_dims.at(2))); // 64x64x3
  canvas.convertTo(canvas, CV_32FC3, 1.0f / 255.0f, 0.f);  // 64x64x3 (0.,1.0)
  // (1,3,64,64)
  ortcv::utils::transform::normalize_inplace(canvas, mean_val, scale_val); // float32

  return ortcv::utils::transform::create_tensor(
      canvas, input_node_dims, memory_info_handler,
      input_values_handler, ortcv::utils::transform::CHW);
}

void SSRNet::detect(const cv::Mat &mat, types::Age &age)
{
  if (mat.empty()) return;
  // 1. make input tensor
  Ort::Value input_tensor = this->transform(mat);
  // 2. inference
  auto output_tensors = ort_session->Run(
      Ort::RunOptions{nullptr}, input_node_names.data(),
      &input_tensor, 1, output_node_names.data(), num_outputs
  );
  Ort::Value &age_tensor = output_tensors.at(0); // (1,)
  const float pred_age = age_tensor.At<float>({0});
  const unsigned int interval_min = static_cast<int>(pred_age - 2.f > 0.f ? pred_age - 2.f : 0.f);
  const unsigned int interval_max = static_cast<int>(pred_age + 3.f < 100.f ? pred_age + 3.f : 100.f);

  age.age = pred_age;
  age.age_interval[0] = interval_min;
  age.age_interval[1] = interval_max;
  age.interval_prob = 1.0f;
  age.flag = true;
}