//
// Created by DefTruth on 2021/11/27.
//

#include "ncnn_efficient_emotion8.h"
#include "lite/utils.h"

using ncnncv::NCNNEfficientEmotion8;

NCNNEfficientEmotion8::NCNNEfficientEmotion8(const std::string &_param_path,
                                             const std::string &_bin_path,
                                             unsigned int _num_threads) :
    BasicNCNNHandler(_param_path, _bin_path, _num_threads)
{
}

void NCNNEfficientEmotion8::transform(const cv::Mat &mat, ncnn::Mat &in)
{
  cv::Mat mat_rs;
  cv::resize(mat, mat_rs, cv::Size(input_width, input_height));
  in = ncnn::Mat::from_pixels(mat_rs.data, ncnn::Mat::PIXEL_BGR2RGB, input_width, input_height);
  in.substract_mean_normalize(mean_vals, norm_vals);
}

void NCNNEfficientEmotion8::detect(const cv::Mat &mat, types::Emotions &emotions)
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
  ncnn::Mat emotion_logits;
  extractor.extract("logits", emotion_logits); // c=1,h=1,w=8
#ifdef LITENCNN_DEBUG
  BasicNCNNHandler::print_shape(emotion_logits, "logits");
#endif

  const unsigned int num_emotions = emotion_logits.w;

  unsigned int pred_label = 0;
  const float *pred_logits_ptr = (float *) emotion_logits.data;

  auto softmax_probs = lite::utils::math::softmax<float>(pred_logits_ptr, num_emotions, pred_label);
  emotions.label = pred_label;
  emotions.score = softmax_probs[pred_label];
  emotions.text = emotion_texts[pred_label];
  emotions.flag = true;
}