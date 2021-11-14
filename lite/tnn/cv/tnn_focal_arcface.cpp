//
// Created by DefTruth on 2021/11/14.
//

#include "tnn_focal_arcface.h"

using tnncv::TNNFocalArcFace;

TNNFocalArcFace::TNNFocalArcFace(const std::string &_proto_path,
                                 const std::string &_model_path,
                                 unsigned int _num_threads) :
    BasicTNNHandler(_proto_path, _model_path, _num_threads)
{
}

void TNNFocalArcFace::transform(const cv::Mat &mat)
{
  cv::Mat canvas;
  cv::resize(mat, canvas, cv::Size(input_width, input_height));
  cv::cvtColor(canvas, canvas, cv::COLOR_BGR2RGB);
  // push into input_mat
  input_mat = std::make_shared<tnn::Mat>(input_device_type, tnn::N8UC3,
                                         input_shape, (void *) canvas.data);
  if (!input_mat->GetData())
  {
#ifdef LITETNN_DEBUG
    std::cout << "input_mat == nullptr! transform failed\n";
#endif
  }
}

void TNNFocalArcFace::detect(const cv::Mat &mat, types::FaceContent &face_content)
{
  if (mat.empty()) return;
  // 1. make input tensor
  this->transform(mat);
  // 2. set input_mat
  tnn::MatConvertParam input_cvt_param;
  input_cvt_param.scale = scale_vals;
  input_cvt_param.bias = bias_vals;

  tnn::Status status;
  status = instance->SetInputMat(input_mat, input_cvt_param);
  if (status != tnn::TNN_OK)
  {
#ifdef LITETNN_DEBUG
    std::cout << "instance->SetInputMat failed!:"
              << status.description().c_str() << "\n";
#endif
    return;
  }

  // 3. forward
  status = instance->Forward();
  if (status != tnn::TNN_OK)
  {
#ifdef LITETNN_DEBUG
    std::cout << "instance->Forward failed!:"
              << status.description().c_str() << "\n";
#endif
    return;
  }

  // 4. fetch output mat
  std::shared_ptr<tnn::Mat> embedding_mat;
  tnn::MatConvertParam embed_cvt_param; // default

  status = instance->GetOutputMat(embedding_mat, embed_cvt_param, "embedding", output_device_type);
  if (status != tnn::TNN_OK)
  {
#ifdef LITETNN_DEBUG
    std::cout << "instance->GetOutputMat failed!:"
              << status.description().c_str() << "\n";
#endif
    return;
  }
  auto embedding_dims = embedding_mat->GetDims(); // (1,512)
  const unsigned int hidden_dim = embedding_dims.at(1);
  const float *embedding_values = (float *) embedding_mat->GetData();

  std::vector<float> embedding_norm(embedding_values, embedding_values + hidden_dim);
  cv::normalize(embedding_norm, embedding_norm); // l2 normalize
  face_content.embedding.assign(embedding_norm.begin(), embedding_norm.end());
  face_content.dim = hidden_dim;
  face_content.flag = true;
}


