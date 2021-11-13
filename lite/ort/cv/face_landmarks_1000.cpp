//
// Created by DefTruth on 2021/7/28.
//

#include "face_landmarks_1000.h"
#include "lite/ort/core/ort_utils.h"

using ortcv::FaceLandmark1000;

Ort::Value FaceLandmark1000::transform(const cv::Mat &mat)
{
  cv::Mat canvas;
  cv::resize(mat, canvas, cv::Size(input_node_dims.at(3),
                                   input_node_dims.at(2)));
  cv::cvtColor(canvas, canvas, cv::COLOR_BGR2GRAY); // (1,1,128,128)
  return ortcv::utils::transform::create_tensor(
      canvas, input_node_dims, memory_info_handler,
      input_values_handler, ortcv::utils::transform::CHW);
}

void FaceLandmark1000::detect(const cv::Mat &mat, types::Landmarks &landmarks)
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
  Ort::Value &landmarks_norm = output_tensors.at(0); // (1,1953)
  auto landmark_dims = output_node_dims.at(0);
  unsigned int num_landmarks = landmark_dims.at(1);
  if (num_landmarks > 1946) num_landmarks = 1946;

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