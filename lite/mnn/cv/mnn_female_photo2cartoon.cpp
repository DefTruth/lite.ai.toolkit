//
// Created by DefTruth on 2022/6/12.
//

#include "mnn_female_photo2cartoon.h"

using mnncv::MNNFemalePhoto2Cartoon;

MNNFemalePhoto2Cartoon::MNNFemalePhoto2Cartoon(const std::string &_mnn_path, unsigned int _num_threads)
    : BasicMNNHandler(_mnn_path, _num_threads)
{
  initialize_pretreat();
}

inline void MNNFemalePhoto2Cartoon::initialize_pretreat()
{
  pretreat = std::shared_ptr<MNN::CV::ImageProcess>(
      MNN::CV::ImageProcess::create(
          MNN::CV::BGR,
          MNN::CV::RGB,
          mean_vals, 3,
          norm_vals, 3
      )
  );
}

void MNNFemalePhoto2Cartoon::transform(const cv::Mat &mat_merged_rs)
{
  // (1,3,256,256) deepcopy inside
  pretreat->convert(mat_merged_rs.data, input_width, input_height, mat_merged_rs.step[0], input_tensor);
}

void MNNFemalePhoto2Cartoon::detect(
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
  cv::Mat mat_merged_rs = mat_rs.mul(mask_rs) + (1.f - mask_rs) * 255.f;
  mat_merged_rs.convertTo(mat_merged_rs, CV_8UC3); // keep CV_8UC3 BGR

  // 1. make input tensor
  this->transform(mat_merged_rs);
  // 2. inference cartoon (1,3,256,256)
  mnn_interpreter->runSession(mnn_session);
  auto output_tensors = mnn_interpreter->getSessionOutputAll(mnn_session);
  // 3. generate cartoon
  this->generate_cartoon(output_tensors, mask_rs, content);
}

void MNNFemalePhoto2Cartoon::generate_cartoon(
    const std::map<std::string, MNN::Tensor *> &output_tensors,
    const cv::Mat &mask_rs, types::FemalePhoto2CartoonContent &content)
{
  auto device_cartoon_pred = output_tensors.at("output");
  MNN::Tensor host_cartoon_tensor(device_cartoon_pred, device_cartoon_pred->getDimensionType());
  device_cartoon_pred->copyToHostTensor(&host_cartoon_tensor);

  auto cartoon_dims = host_cartoon_tensor.shape();
  const unsigned int out_h = cartoon_dims.at(2);
  const unsigned int out_w = cartoon_dims.at(3);
  const unsigned int channel_step = out_h * out_w;
  const unsigned int mask_h = mask_rs.rows;
  const unsigned int mask_w = mask_rs.cols;
  // fast assign & channel transpose(CHW->HWC).
  float *cartoon_ptr = host_cartoon_tensor.host<float>();
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
  cv::merge(cartoon_channel_mats, cartoon); // CV_32FC3 allocated
  if (out_h != mask_h || out_w != mask_w)
    cv::resize(cartoon, cartoon, cv::Size(mask_w, mask_h));
  // combine & RGB -> BGR -> uint8
  cartoon = cartoon.mul(mask_rs) + (1.f - mask_rs) * 255.f;
  cv::cvtColor(cartoon, cartoon, cv::COLOR_RGB2BGR);
  cartoon.convertTo(cartoon, CV_8UC3);

  content.cartoon = cartoon;
  content.flag = true;
}
































