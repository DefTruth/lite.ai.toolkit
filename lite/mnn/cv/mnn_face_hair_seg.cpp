//
// Created by DefTruth on 2022/6/19.
//

#include "mnn_face_hair_seg.h"
#include "lite/utils.h"

using mnncv::MNNFaceHairSeg;

MNNFaceHairSeg::MNNFaceHairSeg(const std::string &_mnn_path, unsigned int _num_threads)
    : BasicMNNHandler(_mnn_path, _num_threads)
{
  initialize_pretreat();
}

void MNNFaceHairSeg::initialize_pretreat()
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

void MNNFaceHairSeg::transform(const cv::Mat &mat)
{
  cv::Mat canvas;
  cv::resize(mat, canvas, cv::Size(input_width, input_height));
  // (1,3,224,224) deepcopy inside
  pretreat->convert(canvas.data, input_width, input_height, canvas.step[0], input_tensor);
}

void MNNFaceHairSeg::detect(const cv::Mat &mat, types::FaceHairSegContent &content,
                            bool remove_noise)
{
  if (mat.empty()) return;
  // 1. make input tensor
  this->transform(mat);
  // 2. inference
  mnn_interpreter->runSession(mnn_session);
  auto output_tensors = mnn_interpreter->getSessionOutputAll(mnn_session);
  // 3. generate mask
  this->generate_mask(output_tensors, mat, content, remove_noise);
}

static inline float argmax(float *mutable_ptr, const unsigned int &step)
{
  std::vector<float> logits(3, 0.f);
  logits[0] = *mutable_ptr; // background
  logits[1] = *(mutable_ptr + step); // face
  logits[2] = *(mutable_ptr + 2 * step); // hair
  float label = 0.f;
  float max_logit = logits[0];
  for (unsigned int i = 1; i < 3; ++i)
  {
    if (logits[i] > max_logit)
    {
      max_logit = logits[i];
      label = (float) i;
    }
  }
  // normalize -> 0.~1.
  return label / 2.f; // 0. bgr 0.5 face 1. hair
}

void MNNFaceHairSeg::generate_mask(const std::map<std::string, MNN::Tensor *> &output_tensors,
                                   const cv::Mat &mat, types::FaceHairSegContent &content,
                                   bool remove_noise)
{
  auto device_output_ptr = output_tensors.at("output"); // e.g (1,3,224,224)
  MNN::Tensor host_output_tensor(device_output_ptr, device_output_ptr->getDimensionType());
  device_output_ptr->copyToHostTensor(&host_output_tensor);
  const unsigned int h = mat.rows;
  const unsigned int w = mat.cols;

  auto output_dims = host_output_tensor.shape();
  const unsigned int out_h = output_dims.at(2);
  const unsigned int out_w = output_dims.at(3);
  const unsigned int channel_step = out_h * out_w;

  float *output_ptr = host_output_tensor.host<float>();

  std::vector<float> elements(channel_step, 0.f); // allocate
  for (unsigned int i = 0; i < channel_step; ++i)
    elements[i] = (float) argmax(output_ptr + i, channel_step); // with normalize

  cv::Mat mask(out_h, out_w, CV_32FC1, elements.data());
  // post process
  if (remove_noise) lite::utils::remove_small_connected_area(mask, 0.05f);
  // already allocated a new continuous memory after resize.
  if (out_h != h || out_w != w) cv::resize(mask, mask, cv::Size(w, h));
    // need clone to allocate a new continuous memory if not performed resize.
    // The memory elements point to will release after return.
  else mask = mask.clone();

  content.mask = mask; // auto handle the memory inside ocv with smart ref.
  content.flag = true;
}