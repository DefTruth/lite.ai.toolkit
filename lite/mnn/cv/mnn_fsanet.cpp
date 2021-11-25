//
// Created by DefTruth on 2021/11/25.
//

#include "mnn_fsanet.h"

using mnncv::MNNFSANet;

MNNFSANet::MNNFSANet(const std::string &_mnn_path, unsigned int _num_threads)
    : BasicMNNHandler(_mnn_path, _num_threads)
{
  initialize_pretreat();
}

inline void MNNFSANet::initialize_pretreat()
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

void MNNFSANet::transform(const cv::Mat &mat)
{
  cv::Mat canvas;
  // 0. padding
  const int h = mat.rows;
  const int w = mat.cols;
  const int nh = static_cast<int>((static_cast<float>(h) + pad * static_cast<float>(h)));
  const int nw = static_cast<int>((static_cast<float>(w) + pad * static_cast<float>(w)));

  const int nx1 = std::max(0, static_cast<int>((nw - w) / 2));
  const int ny1 = std::max(0, static_cast<int>((nh - h) / 2));

  canvas = cv::Mat(nh, nw, CV_8UC3, cv::Scalar(0, 0, 0));
  mat.copyTo(canvas(cv::Rect(nx1, ny1, w, h)));
  cv::resize(canvas, canvas, cv::Size(input_width, input_height));

  pretreat->convert(canvas.data, input_width, input_height, canvas.step[0], input_tensor);
}

void MNNFSANet::detect(const cv::Mat &mat, types::EulerAngles &euler_angles)
{
  if (mat.empty()) return;

  // 1. make input tensor
  this->transform(mat);
  // 2. inference
  mnn_interpreter->runSession(mnn_session);
  auto output_tensors = mnn_interpreter->getSessionOutputAll(mnn_session);
  // 3. fetch angles.
  auto device_angles_ptr = output_tensors.at("output"); // (1,3)
  MNN::Tensor host_angles_tensor(device_angles_ptr, device_angles_ptr->getDimensionType());
  device_angles_ptr->copyToHostTensor(&host_angles_tensor);

  const float *angles = host_angles_tensor.host<float>();

  euler_angles.yaw = angles[0];
  euler_angles.pitch = angles[1];
  euler_angles.roll = angles[2];
  euler_angles.flag = true;
}