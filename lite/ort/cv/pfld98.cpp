//
// Created by DefTruth on 2021/7/26.
//

#include "pfld98.h"
#include "lite/ort/core/ort_utils.h"

using ortcv::PFLD98;

Ort::Value PFLD98::transform(const cv::Mat &mat)
{
  cv::Mat canvas;
  cv::resize(mat, canvas, cv::Size(input_node_dims.at(3),
                                   input_node_dims.at(2)));
  // (1,3,112,112)
  ortcv::utils::transform::normalize_inplace(canvas, mean_val, scale_val); // float32

  return ortcv::utils::transform::create_tensor(
      canvas, input_node_dims, memory_info_handler,
      input_values_handler, ortcv::utils::transform::CHW);
}

void PFLD98::detect(const cv::Mat &mat, types::Landmarks &landmarks)
{
  if (mat.empty()) return;
  // this->transform(mat);
  float img_height = static_cast<float>(mat.rows);
  float img_width = static_cast<float>(mat.cols);

  // 1. make input tensor
  Ort::Value input_tensor = this->transform(mat);
  // 2. inference
  auto output_tensors = ort_session->Run(
      Ort::RunOptions{nullptr}, input_node_names.data(),
      &input_tensor, 1, output_node_names.data(), num_outputs
  );
  // 3. fetch landmarks.
  Ort::Value &landmarks_norm = output_tensors.at(1); // (1,98*2)
  auto landmark_dims = output_node_dims.at(1);
  const unsigned int num_landmarks = landmark_dims.at(1);

  for (unsigned int i = 0; i < num_landmarks; i += 2)
  {
    float x = landmarks_norm.At<float>({0, i});
    float y = landmarks_norm.At<float>({0, i + 1});

    x = std::min(std::max(0.f, x), 1.0f);
    y = std::min(std::max(0.f, y), 1.0f);

    landmarks.points.push_back(cv::Point2f(x * img_width, y * img_height));
  }
  landmarks.flag = true;
}