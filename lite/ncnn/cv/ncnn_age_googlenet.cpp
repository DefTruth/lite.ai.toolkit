//
// Created by DefTruth on 2021/11/27.
//

#include "ncnn_age_googlenet.h"
#include "lite/utils.h"

using ncnncv::NCNNAgeGoogleNet;

NCNNAgeGoogleNet::NCNNAgeGoogleNet(const std::string &_param_path,
                                   const std::string &_bin_path,
                                   unsigned int _num_threads) :
    BasicNCNNHandler(_param_path, _bin_path, _num_threads)
{
}

void NCNNAgeGoogleNet::transform(const cv::Mat &mat, ncnn::Mat &in)
{
  cv::Mat mat_rs;
  // (1,3,224,224)
  cv::resize(mat, mat_rs, cv::Size(input_width, input_height));
  in = ncnn::Mat::from_pixels(mat_rs.data, ncnn::Mat::PIXEL_BGR2RGB, input_width, input_height);
  in.substract_mean_normalize(mean_vals, norm_vals);
}

void NCNNAgeGoogleNet::detect(const cv::Mat &mat, types::Age &age)
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
  // 3. fetch.
  ncnn::Mat age_logits;
  extractor.extract("loss3/loss3_Y", age_logits); // c=1,h=1,w=8
#ifdef LITENCNN_DEBUG
  BasicNCNNHandler::print_shape(age_logits, "loss3/loss3_Y");
#endif

  unsigned int interval = 0;
  const unsigned int num_intervals = age_logits.w; // 8
  const float *pred_logits_ptr = (float *) age_logits.data;

  auto softmax_probs = lite::utils::math::softmax<float>(pred_logits_ptr, num_intervals, interval);
  const float pred_age = static_cast<float>(age_intervals[interval][0] + age_intervals[interval][1]) / 2.0f;

  age.age = pred_age;
  age.age_interval[0] = age_intervals[interval][0];
  age.age_interval[1] = age_intervals[interval][1];
  age.interval_prob = softmax_probs[interval];
  age.flag = true;
}