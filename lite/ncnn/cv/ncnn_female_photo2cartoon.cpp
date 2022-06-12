//
// Created by DefTruth on 2022/6/12.
//

#include "ncnn_female_photo2cartoon.h"

using ncnncv::NCNNFemalePhoto2Cartoon;

NCNNFemalePhoto2Cartoon::NCNNFemalePhoto2Cartoon(
    const std::string &_param_path,
    const std::string &_bin_path,
    unsigned int _num_threads,
    unsigned int _input_height,
    unsigned int _input_width) :
    BasicNCNNHandler(_param_path, _bin_path, _num_threads),
    input_height(_input_height), input_width(_input_width)
{
}

void NCNNFemalePhoto2Cartoon::transform(const cv::Mat &mat_merged_rs, ncnn::Mat &in)
{
  // will do deepcopy inside ncnn
  in = ncnn::Mat::from_pixels(mat_merged_rs.data, ncnn::Mat::PIXEL_BGR2RGB, input_width, input_height);
  in.substract_mean_normalize(mean_vals, norm_vals);
}

void NCNNFemalePhoto2Cartoon::detect(
    const cv::Mat &mat, const cv::Mat &mask,
    types::FemalePhoto2CartoonContent &content)
{
  if (mat.empty() || mask.empty()) return;
  const unsigned int channels = mat.channels();
  if (channels != 3) return;
  const unsigned int mask_channels = mask.channels();
  if (mask_channels != 1 && mask_channels != 3) return;
  // model input size
  const unsigned int input_h = input_height; // 256
  const unsigned int input_w = input_width; // 256

  // resize before merging mat and mask
  cv::Mat mat_rs, mask_rs;
  cv::resize(mat, mat_rs, cv::Size(input_w, input_h));
  cv::resize(mask, mask_rs, cv::Size(input_w, input_h)); // CV_32FC1
  if (mask_channels != 3) cv::cvtColor(mask_rs, mask_rs, cv::COLOR_GRAY2BGR); // CV_32FC3
  mat_rs.convertTo(mat_rs, CV_32FC3, 1.f, 0.f); // CV_32FC3
  // merge mat_rs and mask_rs
  cv::Mat mat_merged_rs = mat_rs.mul(mask_rs) + (1.f - mask_rs) * 255.f;
  mat_merged_rs.convertTo(mat_merged_rs, CV_8UC3); // keep CV_8UC3 BGR
  // 1. make input tensor
  ncnn::Mat input;
  this->transform(mat_merged_rs, input);
  // 2. inference & extract
  auto extractor = net->create_extractor();
  extractor.set_light_mode(false);  // default
  extractor.set_num_threads(num_threads);
  extractor.input("input", input);
  // 3. generate cartoon
  this->generate_cartoon(extractor, mask_rs, content);
}

void NCNNFemalePhoto2Cartoon::generate_cartoon(
    ncnn::Extractor &extractor, const cv::Mat &mask_rs,
    types::FemalePhoto2CartoonContent &content)
{
  ncnn::Mat cartoon_pred;
  extractor.extract("output", cartoon_pred);
#ifdef LITENCNN_DEBUG
  BasicNCNNHandler::print_shape(cartoon_pred, "output");
#endif

  const unsigned int out_h = cartoon_pred.h;
  const unsigned int out_w = cartoon_pred.w;
  const unsigned int channel_step = out_h * out_w;
  const unsigned int mask_h = mask_rs.rows;
  const unsigned int mask_w = mask_rs.cols;
  // fast assign & channel transpose(CHW->HWC).
  float *cartoon_ptr = (float *) cartoon_pred.data;
  std::vector<cv::Mat> cartoon_channel_mats;
  cv::Mat rmat(out_h, out_w, CV_32FC1, cartoon_ptr); // R
  cv::Mat gmat(out_h, out_w, CV_32FC1, cartoon_ptr + channel_step); // G
  cv::Mat bmat(out_h, out_w, CV_32FC1, cartoon_ptr + 2 * channel_step); // B
  rmat = (rmat + 1.f) * 127.5f;
  gmat = (gmat + 1.f) * 127.5f;
  bmat = (bmat + 1.f) * 127.5f;
  cartoon_channel_mats.push_back(rmat);
  cartoon_channel_mats.push_back(gmat);
  cartoon_channel_mats.push_back(bmat);
  cv::Mat cartoon;
  cv::merge(cartoon_channel_mats, cartoon); // CV_32FC3
  if (out_h != mask_h || out_w != mask_w)
    cv::resize(cartoon, cartoon, cv::Size(mask_w, mask_h));
  // combine & RGB -> BGR -> uint8
  cartoon = cartoon.mul(mask_rs) + (1.f - mask_rs) * 255.f;
  cv::cvtColor(cartoon, cartoon, cv::COLOR_RGB2BGR);
  cartoon.convertTo(cartoon, CV_8UC3);

  content.cartoon = cartoon;
  content.flag = true;
}
