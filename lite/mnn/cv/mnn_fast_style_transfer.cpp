//
// Created by DefTruth on 2021/11/29.
//

#include "mnn_fast_style_transfer.h"

using mnncv::MNNFastStyleTransfer;

MNNFastStyleTransfer::MNNFastStyleTransfer(const std::string &_mnn_path, unsigned int _num_threads)
    : BasicMNNHandler(_mnn_path, _num_threads)
{
  initialize_pretreat();
}

inline void MNNFastStyleTransfer::initialize_pretreat()
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

void MNNFastStyleTransfer::transform(const cv::Mat &mat)
{
  cv::Mat canvas;
  cv::resize(mat, canvas, cv::Size(input_width, input_height));
  // (1,3,224,224)
  pretreat->convert(canvas.data, input_width, input_height, canvas.step[0], input_tensor);
}

void MNNFastStyleTransfer::detect(const cv::Mat &mat, types::StyleContent &style_content)
{
  if (mat.empty()) return;
  // 1. make input tensor
  this->transform(mat);
  // 2. inference
  mnn_interpreter->runSession(mnn_session);
  auto output_tensors = mnn_interpreter->getSessionOutputAll(mnn_session);
  // 3. fetch.
  auto device_pred_ptr = output_tensors.at("output1");
  MNN::Tensor host_pred_tensor(device_pred_ptr, device_pred_ptr->getDimensionType());
  device_pred_ptr->copyToHostTensor(&host_pred_tensor);

  auto pred_dims = host_pred_tensor.shape(); // (1,3,224,224)
  const unsigned int rows = pred_dims.at(2); // H
  const unsigned int cols = pred_dims.at(3); // W
  const unsigned int channel_step = rows * cols;

  float *pred_ptr = host_pred_tensor.host<float>();

  // fast copy & channel transpose(CHW->HWC).
  cv::Mat rmat(rows, cols, CV_32FC1, pred_ptr); // ref only, zero-copy.
  cv::Mat gmat(rows, cols, CV_32FC1, pred_ptr + channel_step);
  cv::Mat bmat(rows, cols, CV_32FC1, pred_ptr + 2 * channel_step);
  std::vector<cv::Mat> channel_mats;
  channel_mats.push_back(bmat);
  channel_mats.push_back(gmat);
  channel_mats.push_back(rmat);

  cv::merge(channel_mats, style_content.mat); // BGR

  style_content.mat.convertTo(style_content.mat, CV_8UC3);

  style_content.flag = true;
}