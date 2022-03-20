//
// Created by DefTruth on 2022/3/20.
//

#include "ncnn_pipnet98.h"

using ncnncv::NCNNPIPNet98;

NCNNPIPNet98::NCNNPIPNet98(const std::string &_param_path,
                           const std::string &_bin_path,
                           unsigned int _num_threads) :
    BasicNCNNHandler(_param_path, _bin_path, _num_threads)
{
}

void NCNNPIPNet98::transform(const cv::Mat &mat, ncnn::Mat &in)
{
  cv::Mat mat_rs;
  cv::resize(mat, mat_rs, cv::Size(input_width, input_height));
  // will do deepcopy inside ncnn
  in = ncnn::Mat::from_pixels(mat_rs.data, ncnn::Mat::PIXEL_BGR2RGB, input_width, input_height);
  in.substract_mean_normalize(mean_vals, norm_vals);
}


void NCNNPIPNet98::detect(const cv::Mat &mat, types::Landmarks &landmarks)
{

}

void NCNNPIPNet98::generate_landmarks(types::Landmarks &landmarks,
                                      ncnn::Extractor &extractor,
                                      float img_height, float img_width)
{

}