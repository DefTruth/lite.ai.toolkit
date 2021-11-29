//
// Created by DefTruth on 2021/11/29.
//

#include "ncnn_efficientnet_lite4.h"
#include "lite/utils.h"

using ncnncv::NCNNEfficientNetLite4;

NCNNEfficientNetLite4::NCNNEfficientNetLite4(const std::string &_param_path,
                                             const std::string &_bin_path,
                                             unsigned int _num_threads) :
    BasicNCNNHandler(_param_path, _bin_path, _num_threads)
{
}

void NCNNEfficientNetLite4::transform(const cv::Mat &mat, ncnn::Mat &in)
{
  cv::Mat mat_rs;
  cv::resize(mat, mat_rs, cv::Size(input_width, input_height));
  in = ncnn::Mat::from_pixels(mat_rs.data, ncnn::Mat::PIXEL_BGR2RGB, input_width, input_height);
  in.substract_mean_normalize(mean_vals, norm_vals);
}

void NCNNEfficientNetLite4::detect(const cv::Mat &mat, types::ImageNetContent &content, unsigned int top_k)
{
  if (mat.empty()) return;

  // 1. make input tensor
  ncnn::Mat input;
  this->transform(mat, input);
  // 2. inference & extract
  auto extractor = net->create_extractor();
  extractor.set_light_mode(false);  // default
  extractor.set_num_threads(num_threads);
  extractor.input("images:0", input);
  // 3. fetch.
  ncnn::Mat scores_mat;
  extractor.extract("Softmax:0", scores_mat); // c=1,h=1,w=1000
#ifdef LITENCNN_DEBUG
  BasicNCNNHandler::print_shape(scores_mat, "Softmax:0");
#endif

  const unsigned int num_classes = scores_mat.w;
  const float *scores = (float *) scores_mat.data;

  std::vector<unsigned int> sorted_indices = lite::utils::math::argsort<float>(scores, num_classes);
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