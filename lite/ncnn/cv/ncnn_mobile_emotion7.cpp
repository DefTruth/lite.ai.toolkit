//
// Created by DefTruth on 2021/11/27.
//

#include "ncnn_mobile_emotion7.h"
#include "lite/utils.h"

using ncnncv::NCNNMobileEmotion7;

NCNNMobileEmotion7::NCNNMobileEmotion7(const std::string &_param_path,
                                       const std::string &_bin_path,
                                       unsigned int _num_threads) :
    BasicNCNNHandler(_param_path, _bin_path, _num_threads)
{
}

void NCNNMobileEmotion7::transform(const cv::Mat &mat, ncnn::Mat &in)
{
  cv::Mat mat_rs;
  cv::resize(mat, mat_rs, cv::Size(input_width, input_height));
  in = ncnn::Mat::from_pixels(mat_rs.data, ncnn::Mat::PIXEL_BGR2RGB, input_width, input_height);
  in.substract_mean_normalize(mean_vals, norm_vals);
}

void NCNNMobileEmotion7::detect(const cv::Mat &mat, types::Emotions &emotions)
{
  if (mat.empty()) return;

  // 1. make input tensor
  ncnn::Mat input;
  this->transform(mat, input);
  // 2. inference & extract
  auto extractor = net->create_extractor();
  extractor.set_light_mode(false);  // default
  extractor.set_num_threads(num_threads);
  extractor.input("input_1", input);
  // 3. fetch.
  ncnn::Mat emotion_probs;
  extractor.extract("emotion_preds", emotion_probs); // c=1,h=1,w=7
#ifdef LITENCNN_DEBUG
  BasicNCNNHandler::print_shape(emotion_probs, "emotion_preds");
#endif

  const unsigned int num_emotions = emotion_probs.w;

  unsigned int pred_label = 0;
  const float *pred_probs_ptr = (float *) emotion_probs.data;

  float pred_score = pred_probs_ptr[0];

  for (unsigned int i = 0; i < num_emotions; ++i)
  {
    if (pred_probs_ptr[i] > pred_score)
    {
      pred_score = pred_probs_ptr[i];
      pred_label = i;
    }
  }

  emotions.label = pred_label;
  emotions.score = pred_score;
  emotions.text = emotion_texts[pred_label];
  emotions.flag = true;
}