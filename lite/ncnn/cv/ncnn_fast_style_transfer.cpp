//
// Created by DefTruth on 2021/11/29.
//

#include "ncnn_fast_style_transfer.h"

using ncnncv::NCNNFastStyleTransfer;

NCNNFastStyleTransfer::NCNNFastStyleTransfer(
    const std::string &_param_path,
    const std::string &_bin_path,
    unsigned int _num_threads) :
    BasicNCNNHandler(_param_path, _bin_path, _num_threads)
{
}

void NCNNFastStyleTransfer::transform(const cv::Mat &mat, ncnn::Mat &in)
{
  cv::Mat canvas;
  cv::resize(mat, canvas, cv::Size(input_width, input_height));
  in = ncnn::Mat::from_pixels(canvas.data,
                              ncnn::Mat::PIXEL_BGR2RGB,
                              input_width,
                              input_height);

  in.substract_mean_normalize(mean_vals, norm_vals);
}

void NCNNFastStyleTransfer::detect(const cv::Mat &mat, types::StyleContent &style_content)
{
  if (mat.empty()) return;
  // 1. make input tensor
  ncnn::Mat input;
  this->transform(mat, input);
  // 2. inference & extract
  auto extractor = net->create_extractor();
  extractor.set_light_mode(false);  // default
  extractor.set_num_threads(num_threads);
  extractor.input("input1", input);
  // 3. fetch.
  ncnn::Mat pred;
  extractor.extract("output1", pred); // (1,3,224,224)
#ifdef LITENCNN_DEBUG
  BasicNCNNHandler::print_shape(pred, "output1");
#endif

  const unsigned int rows = pred.h; // H
  const unsigned int cols = pred.w; // W
  const unsigned int channel_step = rows * cols;

  float *pred_ptr = (float *) pred.data;

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