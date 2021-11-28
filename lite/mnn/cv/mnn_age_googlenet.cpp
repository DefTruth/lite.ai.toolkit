//
// Created by DefTruth on 2021/11/27.
//

#include "mnn_age_googlenet.h"
#include "lite/utils.h"

using mnncv::MNNAgeGoogleNet;

MNNAgeGoogleNet::MNNAgeGoogleNet(const std::string &_mnn_path, unsigned int _num_threads)
    : BasicMNNHandler(_mnn_path, _num_threads)
{
  initialize_pretreat();
}

inline void MNNAgeGoogleNet::initialize_pretreat()
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

void MNNAgeGoogleNet::transform(const cv::Mat &mat)
{
  cv::Mat canvas;
  // (1,3,224,224)
  cv::resize(mat, canvas, cv::Size(input_width, input_height));
  pretreat->convert(canvas.data, input_width, input_height, canvas.step[0], input_tensor);
}

void MNNAgeGoogleNet::detect(const cv::Mat &mat, types::Age &age)
{
  if (mat.empty()) return;
  // 1. make input tensor
  this->transform(mat);
  // 2. inference
  mnn_interpreter->runSession(mnn_session);
  auto output_tensors = mnn_interpreter->getSessionOutputAll(mnn_session);
  // 3. fetch age.
  auto device_age_logits_ptr = output_tensors.at("loss3/loss3_Y"); // (1,8)
  MNN::Tensor host_age_logits_tensor(device_age_logits_ptr, device_age_logits_ptr->getDimensionType());
  device_age_logits_ptr->copyToHostTensor(&host_age_logits_tensor);

  auto age_dims = host_age_logits_tensor.shape();
  unsigned int interval = 0;
  const unsigned int num_intervals = age_dims.at(1); // 8
  const float *pred_logits_ptr = host_age_logits_tensor.host<float>();

  auto softmax_probs = lite::utils::math::softmax<float>(pred_logits_ptr, num_intervals, interval);
  const float pred_age = static_cast<float>(age_intervals[interval][0] + age_intervals[interval][1]) / 2.0f;

  age.age = pred_age;
  age.age_interval[0] = age_intervals[interval][0];
  age.age_interval[1] = age_intervals[interval][1];
  age.interval_prob = softmax_probs[interval];
  age.flag = true;
}