//
// Created by DefTruth on 2022/6/12.
//

#include "tnn_female_photo2cartoon.h"

using tnncv::TNNFemalePhoto2Cartoon;

TNNFemalePhoto2Cartoon::TNNFemalePhoto2Cartoon(
    const std::string &_proto_path,
    const std::string &_model_path,
    unsigned int _num_threads) :
    BasicTNNHandler(_proto_path, _model_path, _num_threads)
{
}

void TNNFemalePhoto2Cartoon::transform(const cv::Mat &mat_merged_rs)
{
  // push into input_mat (1,3,256,256)
  input_mat = std::make_shared<tnn::Mat>(input_device_type, tnn::N8UC3,
                                         input_shape, (void *) mat_merged_rs.data);
  if (!input_mat->GetData())
  {
#ifdef LITETNN_DEBUG
    std::cout << "input_mat == nullptr! transform failed\n";
#endif
  }
}

void TNNFemalePhoto2Cartoon::detect(
    const cv::Mat &mat, const cv::Mat &mask,
    types::FemalePhoto2CartoonContent &content)
{
  if (mat.empty() || mask.empty()) return;
  const unsigned int channels = mat.channels();
  if (channels != 3) return;
  const unsigned int mask_channels = mask.channels();
  if (mask_channels != 1 && mask_channels != 3) return;
  // model input size
  const unsigned int input_h = input_height; // 256
  const unsigned int input_w = input_width; // 256
  // resize before merging mat and mask
  cv::Mat mat_rs, mask_rs;
  cv::resize(mat, mat_rs, cv::Size(input_w, input_h));
  cv::resize(mask, mask_rs, cv::Size(input_w, input_h)); // CV_32FC1
  if (mask_channels != 3) cv::cvtColor(mask_rs, mask_rs, cv::COLOR_GRAY2BGR); // CV_32FC3
  mat_rs.convertTo(mat_rs, CV_32FC3, 1.f, 0.f); // CV_32FC3
  // merge mat_rs and mask_rs
  cv::Mat mat_merged_rs = mat_rs.mul(mask_rs) + (1.f - mask_rs) * 255.f; // CV_32FC3
  mat_merged_rs.convertTo(mat_merged_rs, CV_8UC3); // mapping -> tnn::N8UC3
  // 1. make input tensor
  this->transform(mat_merged_rs);
  // 2. set input_mat
  tnn::MatConvertParam input_cvt_param;
  input_cvt_param.scale = scale_vals;
  input_cvt_param.bias = bias_vals;
  input_cvt_param.reverse_channel = true; // BGR -> RGB

  tnn::Status status;
  status = instance->SetInputMat(input_mat, input_cvt_param);
  if (status != tnn::TNN_OK)
  {
#ifdef LITETNN_DEBUG
    std::cout << status.description().c_str() << "\n";
#endif
    return;
  }
  // 3. forward cartoon (1,3,256,256)
  status = instance->Forward();
  if (status != tnn::TNN_OK)
  {
#ifdef LITETNN_DEBUG
    std::cout << status.description().c_str() << "\n";
#endif
    return;
  }
  // 3. generate cartoon
  this->generate_cartoon(instance, mask_rs, content);
}

void TNNFemalePhoto2Cartoon::generate_cartoon(
    std::shared_ptr<tnn::Instance> &_instance,
    const cv::Mat &mask_rs, types::FemalePhoto2CartoonContent &content)
{
  tnn::MatConvertParam cvt_param;
  std::shared_ptr<tnn::Mat> cartoon_pred; // (1,3,256,256)
  auto status = _instance->GetOutputMat(cartoon_pred, cvt_param, "output", output_device_type);
  if (status != tnn::TNN_OK)
  {
#ifdef LITETNN_DEBUG
    std::cout << "instance->GetOutputMat failed!:"
              << status.description().c_str() << "\n";
#endif
    return;
  }

  auto cartoon_dims = cartoon_pred->GetDims();
  const unsigned int out_h = cartoon_dims.at(2);
  const unsigned int out_w = cartoon_dims.at(3);
  const unsigned int channel_step = out_h * out_w;
  const unsigned int mask_h = mask_rs.rows;
  const unsigned int mask_w = mask_rs.cols;
  // fast assign & channel transpose(CHW->HWC).
  float *cartoon_ptr = (float *) cartoon_pred->GetData();
  std::vector<cv::Mat> cartoon_channel_mats;
  cv::Mat rmat(out_h, out_w, CV_32FC1, cartoon_ptr); // R
  cv::Mat gmat(out_h, out_w, CV_32FC1, cartoon_ptr + channel_step); // G
  cv::Mat bmat(out_h, out_w, CV_32FC1, cartoon_ptr + 2 * channel_step); // B
  rmat = (rmat + 1.f) * 127.5f;
  gmat = (gmat + 1.f) * 127.5f;
  bmat = (bmat + 1.f) * 127.5f;
  cartoon_channel_mats.push_back(rmat);
  cartoon_channel_mats.push_back(gmat);
  cartoon_channel_mats.push_back(bmat);
  cv::Mat cartoon;
  cv::merge(cartoon_channel_mats, cartoon); // CV_32FC3
  if (out_h != mask_h || out_w != mask_w)
    cv::resize(cartoon, cartoon, cv::Size(mask_w, mask_h));
  // combine & RGB -> BGR -> uint8
  cartoon = cartoon.mul(mask_rs) + (1.f - mask_rs) * 255.f;
  cv::cvtColor(cartoon, cartoon, cv::COLOR_RGB2BGR);
  cartoon.convertTo(cartoon, CV_8UC3);

  content.cartoon = cartoon;
  content.flag = true;
}























