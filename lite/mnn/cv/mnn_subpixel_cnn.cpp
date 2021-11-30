//
// Created by DefTruth on 2021/11/29.
//

#include "mnn_subpixel_cnn.h"
#include "lite/utils.h"

using mnncv::MNNSubPixelCNN;

MNNSubPixelCNN::MNNSubPixelCNN(const std::string &_mnn_path, unsigned int _num_threads)
    : BasicMNNHandler(_mnn_path, _num_threads)
{
  initialize_pretreat();
}

inline void MNNSubPixelCNN::initialize_pretreat()
{
  pretreat = nullptr; // no use
}

void MNNSubPixelCNN::transform(const cv::Mat &mat)
{
  cv::Mat mat_y; // assume that input mat is Y of YCrCb
  mat.convertTo(mat_y, CV_32FC1, 1.0f / 255.0f, 0.f); // (224,224,1) range (0.,1.0)

  auto tmp_host_nchw_tensor = new MNN::Tensor(input_tensor, MNN::Tensor::CAFFE); // tmp
  std::memcpy(tmp_host_nchw_tensor->host<float>(), mat_y.data,
              input_height * input_width * sizeof(float));
  input_tensor->copyFromHostTensor(tmp_host_nchw_tensor);

  delete tmp_host_nchw_tensor;
}

void MNNSubPixelCNN::detect(const cv::Mat &mat, types::SuperResolutionContent &super_resolution_content)
{
  if (mat.empty()) return;
  cv::Mat mat_copy = mat.clone();
  cv::resize(mat_copy, mat_copy, cv::Size(input_width, input_height)); // (224,224,3)
  cv::Mat mat_ycrcb, mat_y, mat_cr, mat_cb;
  cv::cvtColor(mat_copy, mat_ycrcb, cv::COLOR_BGR2YCrCb);

  // 0. split
  std::vector<cv::Mat> split_mats;
  cv::split(mat_ycrcb, split_mats);
  mat_y = split_mats.at(0); // (224,224,1) uchar CV_8UC1
  mat_cr = split_mats.at(1);
  mat_cb = split_mats.at(2);

  // 1. make input tensor
  this->transform(mat_y); // (1,1,256,256)
  // 2. inference
  mnn_interpreter->runSession(mnn_session);
  auto output_tensors = mnn_interpreter->getSessionOutputAll(mnn_session);

  auto device_pred_ptr = output_tensors.at("output"); // (1,1,672,672)
  MNN::Tensor host_pred_tensor(device_pred_ptr, device_pred_ptr->getDimensionType());
  device_pred_ptr->copyToHostTensor(&host_pred_tensor);

  auto pred_dims = host_pred_tensor.shape();
  const unsigned int rows = pred_dims.at(2); // H
  const unsigned int cols = pred_dims.at(3); // W

  float *pred_ptr = host_pred_tensor.host<float>();

  mat_y = cv::Mat(rows, cols, CV_32FC1, pred_ptr); // release & create

  mat_y *= 255.0f;

  mat_y.convertTo(mat_y, CV_8UC1);

  cv::resize(mat_cr, mat_cr, cv::Size(cols, rows));
  cv::resize(mat_cb, mat_cb, cv::Size(cols, rows));

  std::vector<cv::Mat> out_mats;
  out_mats.push_back(mat_y);
  out_mats.push_back(mat_cr);
  out_mats.push_back(mat_cb);

  // 3. merge
  cv::merge(out_mats, super_resolution_content.mat);
  if (super_resolution_content.mat.empty())
  {
    super_resolution_content.flag = false;
    return;
  }
  cv::cvtColor(super_resolution_content.mat, super_resolution_content.mat, cv::COLOR_YCrCb2BGR);
  super_resolution_content.flag = true;
}