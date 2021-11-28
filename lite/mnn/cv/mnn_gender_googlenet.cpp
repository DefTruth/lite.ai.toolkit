//
// Created by DefTruth on 2021/11/27.
//

#include "mnn_gender_googlenet.h"
#include "lite/utils.h"

using mnncv::MNNGenderGoogleNet;

MNNGenderGoogleNet::MNNGenderGoogleNet(const std::string &_mnn_path, unsigned int _num_threads)
    : BasicMNNHandler(_mnn_path, _num_threads)
{
  initialize_pretreat();
}

inline void MNNGenderGoogleNet::initialize_pretreat()
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

void MNNGenderGoogleNet::transform(const cv::Mat &mat)
{
  cv::Mat canvas;
  // (1,3,224,224)
  cv::resize(mat, canvas, cv::Size(input_width, input_height));
  pretreat->convert(canvas.data, input_width, input_height, canvas.step[0], input_tensor);
}

void MNNGenderGoogleNet::detect(const cv::Mat &mat, types::Gender &gender)
{
  if (mat.empty()) return;
  // 1. make input tensor
  this->transform(mat);
  // 2. inference
  mnn_interpreter->runSession(mnn_session);
  auto output_tensors = mnn_interpreter->getSessionOutputAll(mnn_session);
  // 3. fetch.
  auto device_gender_logits_ptr = output_tensors.at("loss3/loss3_Y"); // (1,2)
  MNN::Tensor host_gender_logits_tensor(device_gender_logits_ptr, device_gender_logits_ptr->getDimensionType());
  device_gender_logits_ptr->copyToHostTensor(&host_gender_logits_tensor);

  auto gender_dims = host_gender_logits_tensor.shape();
  const unsigned int num_genders = gender_dims.at(1); // 2
  const float *pred_logits_ptr = host_gender_logits_tensor.host<float>();

  unsigned int pred_gender = 0;
  auto softmax_probs = lite::utils::math::softmax<float>(pred_logits_ptr, num_genders, pred_gender);
  unsigned int gender_label = pred_gender == 1 ? 0 : 1;
  gender.label = gender_label;
  gender.text = gender_texts[gender_label];
  gender.score = softmax_probs[pred_gender];
  gender.flag = true;
}