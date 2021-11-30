//
// Created by DefTruth on 2021/11/29.
//

#include "ncnn_fcn_resnet101.h"

using ncnncv::NCNNFCNResNet101;

NCNNFCNResNet101::NCNNFCNResNet101(
    const std::string &_param_path,
    const std::string &_bin_path,
    unsigned int _num_threads) :
    BasicNCNNHandler(_param_path, _bin_path, _num_threads)
{
}

void NCNNFCNResNet101::transform(const cv::Mat &mat, ncnn::Mat &in)
{
  const int img_width = mat.cols;
  const int img_height = mat.rows;
  // update dynamic input dims
  dynamic_input_height = img_height;
  dynamic_input_width = img_width;

  in = ncnn::Mat::from_pixels(mat.data,
                              ncnn::Mat::PIXEL_BGR2RGB,
                              dynamic_input_width,
                              dynamic_input_height);

  in.substract_mean_normalize(mean_vals, norm_vals);
}

void NCNNFCNResNet101::detect(const cv::Mat &mat, types::SegmentContent &content)
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
  ncnn::Mat scores;
  extractor.extract("out", scores); // (1,21,h,w) c=21,h,w
#ifdef LITENCNN_DEBUG
  BasicNCNNHandler::print_shape(scores, "out");
#endif

  const unsigned int output_classes = scores.c;
  const unsigned int output_height = scores.h;
  const unsigned int output_width = scores.w;

  const float *scores_ptr = (float *) scores.data;

  // time cost!
  content.names_map.clear();
  content.class_mat = cv::Mat(output_height, output_width, CV_8UC1, cv::Scalar(0));
  content.color_mat = mat.clone();

  const unsigned int scores_step = output_height * output_width; // h x w

  for (unsigned int i = 0; i < output_height; ++i)
  {

    uchar *p_class = content.class_mat.ptr<uchar>(i);
    cv::Vec3b *p_color = content.color_mat.ptr<cv::Vec3b>(i);

    for (unsigned int j = 0; j < output_width; ++j)
    {
      // argmax
      unsigned int max_label = 0;
      float max_conf = scores_ptr[0 * scores_step + i * output_width + j];

      for (unsigned int l = 0; l < output_classes; ++l)
      {
        float conf = scores_ptr[l * scores_step + i * output_width + j];
        if (conf > max_conf)
        {
          max_conf = conf;
          max_label = l;
        }
      }

      if (max_label == 0) continue;

      // assign label for pixel(i,j)
      p_class[j] = cv::saturate_cast<uchar>(max_label);
      // assign color for detected class at pixel(i,j).
      p_color[j][0] = cv::saturate_cast<uchar>((max_label % 10) * 20);
      p_color[j][1] = cv::saturate_cast<uchar>((max_label % 5) * 40);
      p_color[j][2] = cv::saturate_cast<uchar>((max_label % 10) * 20);
      // assign names map
      content.names_map[max_label] = class_names[max_label - 1]; // max_label >= 1
    }

  }

  content.flag = true;
}