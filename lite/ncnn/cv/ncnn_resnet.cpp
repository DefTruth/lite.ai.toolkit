//
// Created by DefTruth on 2021/11/29.
//

#include "ncnn_resnet.h"
#include "lite/utils.h"

using ncnncv::NCNNResNet;

NCNNResNet::NCNNResNet(const std::string &_param_path,
                       const std::string &_bin_path,
                       unsigned int _num_threads) :
    BasicNCNNHandler(_param_path, _bin_path, _num_threads)
{
}

void NCNNResNet::transform(const cv::Mat &mat, ncnn::Mat &in)
{
  cv::Mat mat_rs;
  cv::resize(mat, mat_rs, cv::Size(input_width, input_height));
  in = ncnn::Mat::from_pixels(mat_rs.data, ncnn::Mat::PIXEL_BGR2RGB, input_width, input_height);
  in.substract_mean_normalize(mean_vals, norm_vals);
}

void NCNNResNet::detect(const cv::Mat &mat, types::ImageNetContent &content, unsigned int top_k)
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
  ncnn::Mat logits_mat;
  extractor.extract("logits", logits_mat); // c=1,h=1,w=1000
#ifdef LITENCNN_DEBUG
  BasicNCNNHandler::print_shape(logits_mat, "logits");
#endif

  const unsigned int num_classes = logits_mat.w;
  const float *logits = (float *) logits_mat.data;

  unsigned int max_id;
  std::vector<float> scores = lite::utils::math::softmax<float>(logits, num_classes, max_id);
  std::vector<unsigned int> sorted_indices = lite::utils::math::argsort<float>(scores);
  if (top_k > num_classes) top_k = num_classes;

  content.scores.clear();
  content.labels.clear();
  content.texts.clear();
  for (unsigned int i = 0; i < top_k; ++i)
  {
    content.labels.push_back(sorted_indices[i]);
    content.scores.push_back(scores[sorted_indices[i]]);
    content.texts.push_back(class_names[sorted_indices[i]]);
  }
  content.flag = true;
}
