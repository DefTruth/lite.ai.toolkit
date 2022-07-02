//
// Created by DefTruth on 2022/7/2.
//

#include "ncnn_face_parsing_bisenet.h"

using ncnncv::NCNNFaceParsingBiSeNet;

NCNNFaceParsingBiSeNet::NCNNFaceParsingBiSeNet(const std::string &_param_path,
                                               const std::string &_bin_path,
                                               unsigned int _num_threads,
                                               unsigned int _input_height,
                                               unsigned int _input_width) :
    BasicNCNNHandler(_param_path, _bin_path, _num_threads),
    input_height(_input_height), input_width(_input_width)
{
}

void NCNNFaceParsingBiSeNet::transform(const cv::Mat &mat, ncnn::Mat &in)
{
  cv::Mat mat_rs;
  cv::resize(mat, mat_rs, cv::Size(input_width, input_height));
  // will do deepcopy inside ncnn
  in = ncnn::Mat::from_pixels(mat_rs.data, ncnn::Mat::PIXEL_BGR2RGB, input_width, input_height);
  in.substract_mean_normalize(mean_vals, norm_vals);
}

void NCNNFaceParsingBiSeNet::detect(const cv::Mat &mat, types::FaceParsingContent &content,
                                    bool minimum_post_process)
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
  // 3. generate mask
  this->generate_mask(extractor, mat, content, minimum_post_process);
}

static inline uchar __argmax_find(float *mutable_ptr, const unsigned int &step)
{
  std::vector<float> logits(19, 0.f);
  for (unsigned int i = 0; i < 19; ++i)
    logits[i] = *(mutable_ptr + i * step);
  uchar label = 0;
  float max_logit = logits[0];
  for (unsigned int i = 1; i < 19; ++i)
  {
    if (logits[i] > max_logit)
    {
      max_logit = logits[i];
      label = (uchar) i;
    }
  }
  return label;
}

static const uchar part_colors[20][3] = {
    {255, 0,   0},
    {255, 85,  0},
    {255, 170, 0},
    {255, 0,   85},
    {255, 0,   170},
    {0,   255, 0},
    {85,  255, 0},
    {170, 255, 0},
    {0,   255, 85},
    {0,   255, 170},
    {0,   0,   255},
    {85,  0,   255},
    {170, 0,   255},
    {0,   85,  255},
    {0,   170, 255},
    {255, 255, 0},
    {255, 255, 85},
    {255, 255, 170},
    {255, 0,   255},
    {255, 85,  255}
};

void NCNNFaceParsingBiSeNet::generate_mask(ncnn::Extractor &extractor, const cv::Mat &mat,
                                           types::FaceParsingContent &content,
                                           bool minimum_post_process)
{
  ncnn::Mat output;
  extractor.extract("out", output);
#ifdef LITENCNN_DEBUG
  BasicNCNNHandler::print_shape(output, "out");
#endif
  const unsigned int h = mat.rows;
  const unsigned int w = mat.cols;

  const unsigned int out_h = output.h;
  const unsigned int out_w = output.w;
  const unsigned int channel_step = out_h * out_w;

  float *output_ptr = (float *) output.data;
  std::vector<uchar> elements(channel_step, 0); // allocate
  for (unsigned int i = 0; i < channel_step; ++i)
    elements[i] = __argmax_find(output_ptr + i, channel_step);

  cv::Mat label(out_h, out_w, CV_8UC1, elements.data());

  if (!minimum_post_process)
  {
    const uchar *label_ptr = label.data;
    cv::Mat color_mat(out_h, out_w, CV_8UC3, cv::Scalar(255, 255, 255));
    for (unsigned int i = 0; i < color_mat.rows; ++i)
    {
      cv::Vec3b *p = color_mat.ptr<cv::Vec3b>(i);
      for (unsigned int j = 0; j < color_mat.cols; ++j)
      {
        if (label_ptr[i * out_w + j] == 0) continue;
        p[j][0] = part_colors[label_ptr[i * out_w + j]][0];
        p[j][1] = part_colors[label_ptr[i * out_w + j]][1];
        p[j][2] = part_colors[label_ptr[i * out_w + j]][2];
      }
    }
    if (out_h != h || out_w != w)
      cv::resize(color_mat, color_mat, cv::Size(w, h));
    cv::addWeighted(mat, 0.4, color_mat, 0.6, 0., content.merge);
  }
  if (out_h != h || out_w != w) cv::resize(label, label, cv::Size(w, h));

  content.label = label;
  content.flag = true;
}


















































