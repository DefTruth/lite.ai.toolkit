//
// Created by DefTruth on 2021/11/14.
//

#include "ncnn_cava_ghost_arcface.h"

using ncnncv::NCNNCavaGhostArcFace;

void NCNNCavaGhostArcFace::transform(const cv::Mat &mat, ncnn::Mat &in)
{
  // BGR NHWC -> RGB NCHW
  int h = mat.rows;
  int w = mat.cols;
  in = ncnn::Mat::from_pixels_resize(
      mat.data, ncnn::Mat::PIXEL_BGR2RGB,
      w, h, input_width, input_height
  );
  in.substract_mean_normalize(mean_vals, norm_vals);
}

void NCNNCavaGhostArcFace::detect(const cv::Mat &mat, types::FaceContent &face_content)
{
  if (mat.empty()) return;
  // 1. make input tensor
  ncnn::Mat input;
  this->transform(mat, input);
  // 2. inference & extract
  auto extractor = net->create_extractor();
  extractor.set_light_mode(false);  // default
  extractor.set_num_threads(num_threads);
  extractor.input("input", input);
  ncnn::Mat embedding;
  extractor.extract("embedding", embedding);

  const unsigned int hidden_dim = embedding.w; // 512
  const float *embedding_values = (float *) embedding.data;
  std::vector<float> embedding_norm(embedding_values, embedding_values + hidden_dim);
  cv::normalize(embedding_norm, embedding_norm); // l2 normalize
  face_content.embedding.assign(embedding_norm.begin(), embedding_norm.end());
  face_content.dim = hidden_dim;
  face_content.flag = true;
}