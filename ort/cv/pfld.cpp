//
// Created by DefTruth on 2021/3/14.
//

#include "pfld.h"
#include "ort/core/ort_utils.h"

using ortcv::PFLD;

ort::Value PFLD::transform(const cv::Mat &mat)
{
  cv::Mat canva = mat.clone();
  cv::resize(canva, canva, cv::Size(input_node_dims.at(3),
                                    input_node_dims.at(2)));
  // (1,3,112,112)
  ortcv::utils::transform::normalize_inplace(canva, mean_val, scale_val); // flaot32

  return ortcv::utils::transform::create_tensor(
      canva, input_node_dims, memory_info_handler,
      input_values_handler, ortcv::utils::transform::CHW);
}

void PFLD::detect(const cv::Mat &mat, types::Landmarks &landmarks)
{
  if (mat.empty()) return;
  // this->transform(mat);
  float img_height = static_cast<float>(mat.rows);
  float img_width = static_cast<float>(mat.cols);

  // 1. make input tensor
  ort::Value input_tensor = this->transform(mat);
  // 2. inference
  auto output_tensors = ort_session->Run(
      ort::RunOptions{nullptr}, input_node_names.data(),
      &input_tensor, 1, output_node_names.data(), num_outputs
  );
  // 3. fetch landmarks.
  ort::Value &_landmarks = output_tensors.at(1); // (1,106*2)
  auto landmrk_dims = output_node_dims.at(1);
  const unsigned int num_landmarks = landmrk_dims.at(1);

  for (unsigned int i = 0; i < num_landmarks; i += 2)
  {
    landmarks.points.push_back(
        cv::Point2f(_landmarks.At<float>({0, i}) * img_width,
                    _landmarks.At<float>({0, i + 1}) * img_height
        ));
  }
  landmarks.flag = true;
}