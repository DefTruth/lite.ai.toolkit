//
// Created by DefTruth on 2021/11/21.
//

#include "mnn_pfld98.h"

using mnncv::MNNPFLD98;

MNNPFLD98::MNNPFLD98(const std::string &_mnn_path, unsigned int _num_threads)
    : BasicMNNHandler(_mnn_path, _num_threads)
{
  initialize_pretreat();
}

inline void MNNPFLD98::initialize_pretreat()
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

void MNNPFLD98::transform(const cv::Mat &mat)
{
  cv::Mat mat_rs;
  cv::resize(mat, mat_rs, cv::Size(input_width, input_height));
  pretreat->convert(mat_rs.data, input_width, input_height, mat_rs.step[0], input_tensor);
}

void MNNPFLD98::detect(const cv::Mat &mat, types::Landmarks &landmarks)
{
  if (mat.empty()) return;
  // this->transform(mat);
  float img_height = static_cast<float>(mat.rows);
  float img_width = static_cast<float>(mat.cols);

  // 1. make input tensor
  this->transform(mat);
  // 2. inference
  mnn_interpreter->runSession(mnn_session);
  auto output_tensors = mnn_interpreter->getSessionOutputAll(mnn_session);
  // 3. fetch landmarks.
  auto device_landmarks_ptr = output_tensors.at("landmarks"); // (1,98*2)
  MNN::Tensor host_landmarks_tensor(device_landmarks_ptr, device_landmarks_ptr->getDimensionType());
  device_landmarks_ptr->copyToHostTensor(&host_landmarks_tensor);
  auto landmark_dims = host_landmarks_tensor.shape();

  const unsigned int num_landmarks = landmark_dims.at(1); // (1,98*2)
  const float *landmarks_ptr = host_landmarks_tensor.host<float>();

  for (unsigned int i = 0; i < num_landmarks; i += 2)
  {
    float x = landmarks_ptr[i];
    float y = landmarks_ptr[i + 1];

    x = std::min(std::max(0.f, x), 1.0f);
    y = std::min(std::max(0.f, y), 1.0f);

    landmarks.points.push_back(cv::Point2f(x * img_width, y * img_height));
  }
  landmarks.flag = true;
}
