//
// Created by DefTruth on 2022/7/2.
//

#include "tnn_face_parsing_bisenet.h"

using tnncv::TNNFaceParsingBiSeNet;

TNNFaceParsingBiSeNet::TNNFaceParsingBiSeNet(const std::string &_proto_path,
                                             const std::string &_model_path,
                                             unsigned int _num_threads) :
    BasicTNNHandler(_proto_path, _model_path, _num_threads)
{
}

void TNNFaceParsingBiSeNet::transform(const cv::Mat &mat_rs)
{
  // push into input_mat (1,3,512,512) no deepcopy inside TNN
  input_mat = std::make_shared<tnn::Mat>(input_device_type, tnn::N8UC3,
                                         input_shape, (void *) mat_rs.data);
  if (!input_mat->GetData())
  {
#ifdef LITETNN_DEBUG
    std::cout << "input_mat == nullptr! transform failed\n";
#endif
  }
}

void TNNFaceParsingBiSeNet::detect(const cv::Mat &mat, types::FaceParsingContent &content,
                                   bool minimum_post_process)
{
  // 1. make input mat
  cv::Mat mat_rs;
  cv::resize(mat, mat_rs, cv::Size(input_width, input_height));
  cv::cvtColor(mat_rs, mat_rs, cv::COLOR_BGR2RGB);
  this->transform(mat_rs);
  // 2. set input mat
  tnn::MatConvertParam input_cvt_param;
  input_cvt_param.scale = scale_vals;
  input_cvt_param.bias = bias_vals;

  tnn::Status status;
  status = instance->SetInputMat(input_mat, input_cvt_param);
  if (status != tnn::TNN_OK)
  {
#ifdef LITETNN_DEBUG
    std::cout << status.description().c_str() << "\n";
#endif
    return;
  }
  // 3. forward
  status = instance->Forward();
  if (status != tnn::TNN_OK)
  {
#ifdef LITETNN_DEBUG
    std::cout << status.description().c_str() << "\n";
#endif
    return;
  }
  // 4. generate mask
  this->generate_mask(instance, mat, content, minimum_post_process);
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

void TNNFaceParsingBiSeNet::generate_mask(std::shared_ptr<tnn::Instance> &_instance, const cv::Mat &mat,
                                          types::FaceParsingContent &content,
                                          bool minimum_post_process)
{
  std::shared_ptr<tnn::Mat> output_mat;
  tnn::MatConvertParam cvt_param;
  auto status = _instance->GetOutputMat(output_mat, cvt_param, "out", output_device_type);
  if (status != tnn::TNN_OK)
  {
#ifdef LITETNN_DEBUG
    std::cout << "instance->GetOutputMat failed!:"
              << status.description().c_str() << "\n";
#endif
    return;
  }
  const unsigned int h = mat.rows;
  const unsigned int w = mat.cols;

  auto output_dims = output_mat->GetDims();
  const unsigned int out_h = output_dims.at(2);
  const unsigned int out_w = output_dims.at(3);
  const unsigned int channel_step = out_h * out_w;

  float *output_ptr = (float *) output_mat->GetData();
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



































