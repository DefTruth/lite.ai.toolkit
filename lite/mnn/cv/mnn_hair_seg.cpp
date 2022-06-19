//
// Created by DefTruth on 2022/6/19.
//

#include "mnn_hair_seg.h"
#include "lite/utils.h"

using mnncv::MNNHairSeg;

MNNHairSeg::MNNHairSeg(const std::string &_mnn_path, unsigned int _num_threads)
    : BasicMNNHandler(_mnn_path, _num_threads)
{
  initialize_pretreat();
}

void MNNHairSeg::initialize_pretreat()
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

void MNNHairSeg::transform(const cv::Mat &mat)
{
  cv::Mat canvas;
  cv::resize(mat, canvas, cv::Size(input_width, input_height));
  // (1,3,224,224) deepcopy inside
  pretreat->convert(canvas.data, input_width, input_height, canvas.step[0], input_tensor);
}

void MNNHairSeg::detect(const cv::Mat &mat, types::HairSegContent &content,
                        float score_threshold, bool remove_noise)
{
  if (mat.empty()) return;
  // 1. make input tensor
  this->transform(mat);
  // 2. inference
  mnn_interpreter->runSession(mnn_session);
  auto output_tensors = mnn_interpreter->getSessionOutputAll(mnn_session);
  // 3. generate mask
  this->generate_mask(output_tensors, mat, content, score_threshold, remove_noise);
}

static inline void __zero_if_small_inplace(float *mutable_ptr, float &score)
{ if ((*mutable_ptr) < score) *mutable_ptr = 0.f; }

void MNNHairSeg::generate_mask(const std::map<std::string, MNN::Tensor *> &output_tensors,
                               const cv::Mat &mat, types::HairSegContent &content,
                               float score_threshold, bool remove_noise)
{
  auto device_output_ptr = output_tensors.at("output"); // e.g (1,1,224,224)
  MNN::Tensor host_output_tensor(device_output_ptr, device_output_ptr->getDimensionType());
  device_output_ptr->copyToHostTensor(&host_output_tensor);
  const unsigned int h = mat.rows;
  const unsigned int w = mat.cols;

  auto output_dims = host_output_tensor.shape();
  const unsigned int out_h = output_dims.at(2);
  const unsigned int out_w = output_dims.at(3);
  const unsigned int element_size = out_h * out_w;

  float *output_ptr = host_output_tensor.host<float>();

  // remove small values
  if (score_threshold > 0.001f)
    for (unsigned int i = 0; i < element_size; ++i)
      __zero_if_small_inplace(output_ptr + i, score_threshold);

  cv::Mat mask(out_h, out_w, CV_32FC1, output_ptr);
  // post process
  if (remove_noise) lite::utils::remove_small_connected_area(mask, 0.05f);
  if (out_h != h || out_w != w) cv::resize(mask, mask, cv::Size(w, h));

  content.mask = mask;
  content.flag = true;
}