//
// Created by DefTruth on 2021/11/29.
//

#include "tnn_fast_style_transfer.h"

using tnncv::TNNFastStyleTransfer;

TNNFastStyleTransfer::TNNFastStyleTransfer(const std::string &_proto_path,
                                           const std::string &_model_path,
                                           unsigned int _num_threads) :
    BasicTNNHandler(_proto_path, _model_path, _num_threads)
{
}

void TNNFastStyleTransfer::transform(const cv::Mat &mat_rs)
{
  // push into input_mat
  input_mat = std::make_shared<tnn::Mat>(input_device_type, tnn::N8UC3,
                                         input_shape, (void *) mat_rs.data);
  if (!input_mat->GetData())
  {
#ifdef LITETNN_DEBUG
    std::cout << "input_mat == nullptr! transform failed\n";
#endif
  }
}

void TNNFastStyleTransfer::detect(const cv::Mat &mat, types::StyleContent &style_content)
{
  if (mat.empty()) return;

  // 1. make input mat
  cv::Mat mat_rs;
  cv::resize(mat, mat_rs, cv::Size(input_width, input_height));
  cv::cvtColor(mat_rs, mat_rs, cv::COLOR_BGR2RGB); // (1,224,224,3)
  this->transform(mat_rs);
  // 2. set input_mat
  tnn::MatConvertParam input_cvt_param;
  input_cvt_param.scale = scale_vals;
  input_cvt_param.bias = bias_vals;

  tnn::Status status;
  status = instance->SetInputMat(input_mat, input_cvt_param);
  if (status != tnn::TNN_OK)
  {
#ifdef LITETNN_DEBUG
    std::cout << status.description().c_str() << "\n";
#endif
    return;
  }
  // 3. forward
  status = instance->Forward();
  if (status != tnn::TNN_OK)
  {
#ifdef LITETNN_DEBUG
    std::cout << status.description().c_str() << "\n";
#endif
    return;
  }
  // 4. fetch
  tnn::MatConvertParam cvt_param;
  std::shared_ptr<tnn::Mat> pred_mat; // (1,3,224,224)
  status = instance->GetOutputMat(pred_mat, cvt_param, "output1", output_device_type);

  if (status != tnn::TNN_OK)
  {
#ifdef LITETNN_DEBUG
    std::cout << status.description().c_str() << "\n";
#endif
    return;
  }

  auto pred_dims = pred_mat->GetDims(); // (1,3,224,224)
  const unsigned int rows = pred_dims.at(2); // H
  const unsigned int cols = pred_dims.at(3); // W
  const unsigned int channel_step = rows * cols;

  float *pred_ptr = (float *) pred_mat->GetData();

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