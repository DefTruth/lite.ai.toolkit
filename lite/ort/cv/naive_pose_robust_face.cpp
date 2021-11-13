//
// Created by DefTruth on 2021/7/18.
//

#include "naive_pose_robust_face.h"
#include "lite/ort/core/ort_utils.h"

using ortcv::NaivePoseRobustFace;

Ort::Value NaivePoseRobustFace::transform(const cv::Mat &mat)
{
  cv::Mat canvas;
  cv::resize(mat, canvas, cv::Size(input_node_dims.at(3),
                                   input_node_dims.at(2)));
  cv::cvtColor(canvas, canvas, cv::COLOR_BGR2RGB);
  // (1,3,224,224)
  ortcv::utils::transform::normalize_inplace(canvas, mean_val, scale_val);
  return ortcv::utils::transform::create_tensor(
      canvas, input_node_dims, memory_info_handler,
      input_values_handler, ortcv::utils::transform::CHW);
}

void NaivePoseRobustFace::detect(const cv::Mat &mat, types::FaceContent &face_content)
{
  if (mat.empty()) return;
  // 1. make input tensor
  Ort::Value input_tensor = this->transform(mat);
  // 2. inference
  auto output_tensors = ort_session->Run(
      Ort::RunOptions{nullptr}, input_node_names.data(),
      &input_tensor, 1, output_node_names.data(), num_outputs
  );
  Ort::Value &embedding = output_tensors.at(0);
  auto embedding_dims = output_node_dims.at(0); // (1,256)
  const unsigned int hidden_dim = embedding_dims.at(1); // 256
  const float *embedding_values = embedding.GetTensorMutableData<float>();
  std::vector<float> embedding_norm(embedding_values, embedding_values + hidden_dim);
  cv::normalize(embedding_norm, embedding_norm); // l2 normalize
  face_content.embedding.assign(embedding_norm.begin(), embedding_norm.end());
  face_content.dim = hidden_dim;
  face_content.flag = true;
}