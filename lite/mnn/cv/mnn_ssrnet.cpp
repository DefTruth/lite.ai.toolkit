//
// Created by DefTruth on 2021/11/27.
//

#include "mnn_ssrnet.h"

using mnncv::MNNSSRNet;

MNNSSRNet::MNNSSRNet(const std::string &_mnn_path, unsigned int _num_threads)
    : BasicMNNHandler(_mnn_path, _num_threads)
{
  initialize_pretreat();
}

inline void MNNSSRNet::initialize_pretreat()
{
  pretreat = std::shared_ptr<MNN::CV::ImageProcess>(
      MNN::CV::ImageProcess::create(
          MNN::CV::BGR,
          MNN::CV::BGR,
          mean_vals, 3,
          norm_vals, 3
      )
  );
}

void MNNSSRNet::transform(const cv::Mat &mat)
{
  cv::Mat canvas;
  // (1,3,64,64)
  cv::resize(mat, canvas, cv::Size(input_width, input_height));
  pretreat->convert(canvas.data, input_width, input_height, canvas.step[0], input_tensor);
}

void MNNSSRNet::detect(const cv::Mat &mat, types::Age &age)
{
  if (mat.empty()) return;
  // 1. make input tensor
  this->transform(mat);
  // 2. inference
  mnn_interpreter->runSession(mnn_session);
  auto output_tensors = mnn_interpreter->getSessionOutputAll(mnn_session);
  // 3. fetch.
  auto device_age_ptr = output_tensors.at("age");
  MNN::Tensor host_age_tensor(device_age_ptr, device_age_ptr->getDimensionType());
  device_age_ptr->copyToHostTensor(&host_age_tensor);

  const float *age_ptr = host_age_tensor.host<float>();
  const float pred_age = age_ptr[0];

  const unsigned int interval_min = static_cast<int>(pred_age - 2.f > 0.f ? pred_age - 2.f : 0.f);
  const unsigned int interval_max = static_cast<int>(pred_age + 3.f < 100.f ? pred_age + 3.f : 100.f);

  age.age = pred_age;
  age.age_interval[0] = interval_min;
  age.age_interval[1] = interval_max;
  age.interval_prob = 1.0f;
  age.flag = true;
}