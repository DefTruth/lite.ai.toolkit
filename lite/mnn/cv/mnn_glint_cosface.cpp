//
// Created by DefTruth on 2021/11/13.
//

#include "mnn_glint_cosface.h"

using mnncv::MNNGlintCosFace;

MNNGlintCosFace::MNNGlintCosFace(const std::string &_mnn_path, unsigned int _num_threads) :
    BasicMNNHandler(_mnn_path, _num_threads)
{
  initialize_pretreat();
}


inline void MNNGlintCosFace::initialize_pretreat()
{
  pretreat = std::shared_ptr<MNN::CV::ImageProcess>(
      MNN::CV::ImageProcess::create(
          MNN::CV::BGR,
          MNN::CV::RGB,
          mean_vals, 3,
          norm_vals, 3
      )
  );
}

void MNNGlintCosFace::transform(const cv::Mat &mat)
{
  // normalize & HWC -> CHW & BGR -> RGB
  cv::Mat mat_rs;
  cv::resize(mat, mat_rs, cv::Size(input_width, input_height));
  pretreat->convert(mat_rs.data, input_width, input_height, mat_rs.step[0], input_tensor);
}

void MNNGlintCosFace::detect(const cv::Mat &mat, types::FaceContent &face_content)
{
  if (mat.empty()) return;
  // 1. make input tensor
  this->transform(mat);
  // 2. inference.
  mnn_interpreter->runSession(mnn_session);
  auto output_tensors = mnn_interpreter->getSessionOutputAll(mnn_session);

  auto device_embedding_ptr = output_tensors.at("embedding");
  MNN::Tensor host_embedding_tensor(device_embedding_ptr, device_embedding_ptr->getDimensionType()); // NCHW
  device_embedding_ptr->copyToHostTensor(&host_embedding_tensor);

  auto embedding_dims = host_embedding_tensor.shape(); // (1,512)
  const unsigned int hidden_dim = embedding_dims.at(1);
  const float *embedding_values = host_embedding_tensor.host<float>();

  std::vector<float> embedding_norm(embedding_values, embedding_values + hidden_dim);
  cv::normalize(embedding_norm, embedding_norm); // l2 normalize
  face_content.embedding.assign(embedding_norm.begin(), embedding_norm.end());
  face_content.dim = hidden_dim;
  face_content.flag = true;
}