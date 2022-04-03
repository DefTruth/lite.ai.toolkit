//
// Created by DefTruth on 2022/3/27.
//

#include "ncnn_modnet.h"
#include "lite/utils.h"

using ncnncv::NCNNMODNet;

NCNNMODNet::NCNNMODNet(const std::string &_param_path,
                       const std::string &_bin_path,
                       unsigned int _num_threads,
                       unsigned int _input_height,
                       unsigned int _input_width) :
    BasicNCNNHandler(_param_path, _bin_path, _num_threads),
    input_height(_input_height), input_width(_input_width)
{
}

void NCNNMODNet::transform(const cv::Mat &mat, ncnn::Mat &in)
{
  cv::Mat mat_rs;
  cv::resize(mat, mat_rs, cv::Size(input_width, input_height));
  // will do deepcopy inside ncnn
  in = ncnn::Mat::from_pixels(mat_rs.data, ncnn::Mat::PIXEL_BGR2RGB, input_width, input_height);
  in.substract_mean_normalize(mean_vals, norm_vals);
}

void NCNNMODNet::detect(const cv::Mat &mat, types::MattingContent &content, bool remove_noise)
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
  // 3. generate matting
  this->generate_matting(extractor, mat, content, remove_noise);
}

void NCNNMODNet::generate_matting(ncnn::Extractor &extractor,
                                  const cv::Mat &mat, types::MattingContent &content,
                                  bool remove_noise)
{
  ncnn::Mat output;
  extractor.extract("output", output);
#ifdef LITENCNN_DEBUG
  BasicNCNNHandler::print_shape(output, "output");
#endif
  const unsigned int h = mat.rows;
  const unsigned int w = mat.cols;

  const unsigned int out_h = input_height;
  const unsigned int out_w = input_width;

  float *output_ptr = (float *) output.data;

  cv::Mat alpha_pred(out_h, out_w, CV_32FC1, output_ptr);
  // post process
  if (remove_noise) lite::utils::remove_small_connected_area(alpha_pred, 0.05f);
  // resize alpha
  if (out_h != h || out_w != w)
  {
    cv::resize(alpha_pred, alpha_pred, cv::Size(w, h));
  }
  cv::Mat mat_copy;
  mat.convertTo(mat_copy, CV_32FC3);
  cv::Mat pmat = alpha_pred; // ref

  // merge mat and fgr mat may not need
  std::vector<cv::Mat> mat_channels;
  cv::split(mat_copy, mat_channels);
  cv::Mat bmat = mat_channels.at(0);
  cv::Mat gmat = mat_channels.at(1);
  cv::Mat rmat = mat_channels.at(2); // ref only, zero-copy.
  bmat = bmat.mul(pmat);
  gmat = gmat.mul(pmat);
  rmat = rmat.mul(pmat);
  cv::Mat rest = 1.f - pmat;
  cv::Mat mbmat = bmat.mul(pmat) + rest * 153.f;
  cv::Mat mgmat = gmat.mul(pmat) + rest * 255.f;
  cv::Mat mrmat = rmat.mul(pmat) + rest * 120.f;
  std::vector<cv::Mat> fgr_channel_mats, merge_channel_mats;
  fgr_channel_mats.push_back(bmat);
  fgr_channel_mats.push_back(gmat);
  fgr_channel_mats.push_back(rmat);
  merge_channel_mats.push_back(mbmat);
  merge_channel_mats.push_back(mgmat);
  merge_channel_mats.push_back(mrmat);

  content.pha_mat = pmat;
  cv::merge(fgr_channel_mats, content.fgr_mat);
  cv::merge(merge_channel_mats, content.merge_mat);

  content.fgr_mat.convertTo(content.fgr_mat, CV_8UC3);
  content.merge_mat.convertTo(content.merge_mat, CV_8UC3);

  content.flag = true;
}
