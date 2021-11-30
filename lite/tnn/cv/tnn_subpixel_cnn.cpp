//
// Created by DefTruth on 2021/11/29.
//

#include "tnn_subpixel_cnn.h"

using tnncv::TNNSubPixelCNN;

TNNSubPixelCNN::TNNSubPixelCNN(const std::string &_proto_path,
                               const std::string &_model_path,
                               unsigned int _num_threads) :
    BasicTNNHandler(_proto_path, _model_path, _num_threads)
{
}

void TNNSubPixelCNN::transform(const cv::Mat &mat)
{
  cv::Mat mat_y; // assume that input mat is Y of YCrCb
  mat.convertTo(mat_y, CV_32FC1, 1.0f / 255.0f, 0.f); // (224,224,1) range (0.,1.0)

  input_mat = std::make_shared<tnn::Mat>(input_device_type, tnn::NCHW_FLOAT,
                                         input_shape, (void *) mat_y.data);
  if (!input_mat->GetData())
  {
#ifdef LITETNN_DEBUG
    std::cout << "input_mat == nullptr! transform failed\n";
#endif
  }
}

void TNNSubPixelCNN::detect(const cv::Mat &mat, types::SuperResolutionContent &super_resolution_content)
{
  if (mat.empty()) return;
  cv::Mat mat_copy = mat.clone();
  cv::resize(mat_copy, mat_copy, cv::Size(input_width, input_height)); // (224,224,3)
  cv::Mat mat_ycrcb, mat_y, mat_cr, mat_cb;
  cv::cvtColor(mat_copy, mat_ycrcb, cv::COLOR_BGR2YCrCb);

  // 0. split
  std::vector<cv::Mat> split_mats;
  cv::split(mat_ycrcb, split_mats);
  mat_y = split_mats.at(0); // (224,224,1) uchar CV_8UC1
  mat_cr = split_mats.at(1);
  mat_cb = split_mats.at(2);

  // 1. make input tensor
  this->transform(mat_y); // (1,1,224,224)
  // 2. set input_mat
  tnn::MatConvertParam input_cvt_param;

  tnn::Status status;
  status = instance->SetInputMat(input_mat, input_cvt_param);
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
  // 4. fetch
  tnn::MatConvertParam cvt_param;
  std::shared_ptr<tnn::Mat> pred_mat; // (1,1,672,672)
  status = instance->GetOutputMat(pred_mat, cvt_param, "output", output_device_type);

  if (status != tnn::TNN_OK)
  {
#ifdef LITETNN_DEBUG
    std::cout << status.description().c_str() << "\n";
#endif
    return;
  }

  auto pred_dims = pred_mat->GetDims(); // (1,2,256,256)
  const unsigned int rows = pred_dims.at(2); // H 256
  const unsigned int cols = pred_dims.at(3); // W 256

  float *pred_ptr = (float *) pred_mat->GetData();

  mat_y = cv::Mat(rows, cols, CV_32FC1, pred_ptr); // release & create

  mat_y *= 255.0f;

  mat_y.convertTo(mat_y, CV_8UC1);

  cv::resize(mat_cr, mat_cr, cv::Size(cols, rows));
  cv::resize(mat_cb, mat_cb, cv::Size(cols, rows));

  std::vector<cv::Mat> out_mats;
  out_mats.push_back(mat_y);
  out_mats.push_back(mat_cr);
  out_mats.push_back(mat_cb);

  // 3. merge
  cv::merge(out_mats, super_resolution_content.mat);
  if (super_resolution_content.mat.empty())
  {
    super_resolution_content.flag = false;
    return;
  }
  cv::cvtColor(super_resolution_content.mat, super_resolution_content.mat, cv::COLOR_YCrCb2BGR);
  super_resolution_content.flag = true;
}