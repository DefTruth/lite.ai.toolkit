//
// Created by DefTruth on 2021/11/21.
//

#include "ncnn_mobilenetv2_68.h"

using ncnncv::NCNNMobileNetV268;

NCNNMobileNetV268::NCNNMobileNetV268(const std::string &_param_path,
                                     const std::string &_bin_path,
                                     unsigned int _num_threads) :
    BasicNCNNHandler(_param_path, _bin_path, _num_threads)
{
}

void NCNNMobileNetV268::transform(const cv::Mat &mat, ncnn::Mat &in)
{
  cv::Mat mat_rs;
  cv::resize(mat, mat_rs, cv::Size(input_width, input_height));
  in = ncnn::Mat::from_pixels(mat_rs.data, ncnn::Mat::PIXEL_BGR2RGB, input_width, input_height);
  in.substract_mean_normalize(mean_vals, norm_vals);
}

void NCNNMobileNetV268::detect(const cv::Mat &mat, types::Landmarks &landmarks)
{
  if (mat.empty()) return;
  float img_height = static_cast<float>(mat.rows);
  float img_width = static_cast<float>(mat.cols);

  // 1. make input tensor
  ncnn::Mat input;
  this->transform(mat, input);
  // 2. inference & extract
  auto extractor = net->create_extractor();
  extractor.set_light_mode(false);  // default
  extractor.set_num_threads(num_threads);
  extractor.input("input", input);
  // 3. fetch landmarks.
  ncnn::Mat landmarks_norm;
  extractor.extract("output", landmarks_norm); // c=1,w=68*2,h=1
#ifdef LITENCNN_DEBUG
  BasicNCNNHandler::print_shape(landmarks_norm, "output");
#endif
  const unsigned int num_landmarks = landmarks_norm.w;
  const float *landmarks_ptr = (float *) landmarks_norm.data;

  for (unsigned int i = 0; i < num_landmarks; i += 2)
  {
    float x = landmarks_ptr[i];
    float y = landmarks_ptr[i + 1];

    x = std::min(std::max(0.f, x), 1.0f);
    y = std::min(std::max(0.f, y), 1.0f);

    landmarks.points.push_back(cv::Point2f(x * img_width, y * img_height));
  }
  landmarks.flag = true;
}