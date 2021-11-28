//
// Created by DefTruth on 2021/11/27.
//

#include "ncnn_gender_googlenet.h"
#include "lite/utils.h"

using ncnncv::NCNNGenderGoogleNet;

NCNNGenderGoogleNet::NCNNGenderGoogleNet(const std::string &_param_path,
                                         const std::string &_bin_path,
                                         unsigned int _num_threads) :
    BasicNCNNHandler(_param_path, _bin_path, _num_threads)
{
}

void NCNNGenderGoogleNet::transform(const cv::Mat &mat, ncnn::Mat &in)
{
  cv::Mat mat_rs;
  // (1,3,224,224)
  cv::resize(mat, mat_rs, cv::Size(input_width, input_height));
  in = ncnn::Mat::from_pixels(mat_rs.data, ncnn::Mat::PIXEL_BGR2RGB, input_width, input_height);
  in.substract_mean_normalize(mean_vals, norm_vals);
}

void NCNNGenderGoogleNet::detect(const cv::Mat &mat, types::Gender &gender)
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
  ncnn::Mat gender_logits;
  extractor.extract("loss3/loss3_Y", gender_logits); // c=1,h=1,w=2
#ifdef LITENCNN_DEBUG
  BasicNCNNHandler::print_shape(gender_logits, "loss3/loss3_Y");
#endif

  const unsigned int num_genders = gender_logits.w;
  const float *pred_logits_ptr = (float *) gender_logits.data;

  unsigned int pred_gender = 0;
  auto softmax_probs = lite::utils::math::softmax<float>(pred_logits_ptr, num_genders, pred_gender);
  unsigned int gender_label = pred_gender == 1 ? 0 : 1;
  gender.label = gender_label;
  gender.text = gender_texts[gender_label];
  gender.score = softmax_probs[pred_gender];
  gender.flag = true;
}