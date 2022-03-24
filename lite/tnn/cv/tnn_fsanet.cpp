//
// Created by DefTruth on 2021/11/25.
//

#include "tnn_fsanet.h"

using tnncv::TNNFSANet;

TNNFSANet::TNNFSANet(const std::string &_proto_path,
                     const std::string &_model_path,
                     unsigned int _num_threads) :
    BasicTNNHandler(_proto_path, _model_path, _num_threads)
{
}

void TNNFSANet::transform(const cv::Mat &mat_padded)
{
  // push into input_mat
  // be carefully, no deepcopy inside this tnn::Mat constructor,
  // so, we can not pass a local cv::Mat to this constructor.
  input_mat = std::make_shared<tnn::Mat>(input_device_type, tnn::N8UC3,
                                         input_shape, (void *) mat_padded.data);
  if (!input_mat->GetData())
  {
#ifdef LITETNN_DEBUG
    std::cout << "input_mat == nullptr! transform failed\n";
#endif
  }
}

void TNNFSANet::detect(const cv::Mat &mat, types::EulerAngles &euler_angles)
{
  if (mat.empty()) return;

  // 1. make input mat
  cv::Mat mat_padded;
  // 0. padding
  const int h = mat.rows;
  const int w = mat.cols;
  const int nh = static_cast<int>((static_cast<float>(h) + pad * static_cast<float>(h)));
  const int nw = static_cast<int>((static_cast<float>(w) + pad * static_cast<float>(w)));

  const int nx1 = std::max(0, static_cast<int>((nw - w) / 2));
  const int ny1 = std::max(0, static_cast<int>((nh - h) / 2));

  mat_padded = cv::Mat(nh, nw, CV_8UC3, cv::Scalar(0, 0, 0));
  mat.copyTo(mat_padded(cv::Rect(nx1, ny1, w, h)));
  cv::resize(mat_padded, mat_padded, cv::Size(input_width, input_height));

  this->transform(mat_padded);
  // 2. set input_mat
  tnn::MatConvertParam input_cvt_param;
  input_cvt_param.scale = scale_vals;
  input_cvt_param.bias = bias_vals;

  tnn::Status status;
  status = instance->SetInputMat(input_mat, input_cvt_param, "input");
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
  // 4. fetch angles.
  tnn::MatConvertParam cvt_param;
  std::shared_ptr<tnn::Mat> angles; // (1,3)
  status = instance->GetOutputMat(angles, cvt_param, "output", output_device_type);

  if (status != tnn::TNN_OK)
  {
#ifdef LITETNN_DEBUG
    std::cout << status.description().c_str() << "\n";
#endif
    return;
  }

  const float *angles_ptr = (float *) angles->GetData();

  euler_angles.yaw = angles_ptr[0];
  euler_angles.pitch = angles_ptr[1];
  euler_angles.roll = angles_ptr[2];
  euler_angles.flag = true;
}