//
// Created by DefTruth on 2021/11/27.
//

#include "tnn_ssrnet.h"

using tnncv::TNNSSRNet;

TNNSSRNet::TNNSSRNet(const std::string &_proto_path,
                     const std::string &_model_path,
                     unsigned int _num_threads) :
    BasicTNNHandler(_proto_path, _model_path, _num_threads)
{
}

void TNNSSRNet::transform(const cv::Mat &mat)
{
  cv::Mat mat_rs;
  cv::resize(mat, mat_rs, cv::Size(input_width, input_height));
  // push into input_mat (1,3,64,64)
  input_mat = std::make_shared<tnn::Mat>(input_device_type, tnn::N8UC3,
                                         input_shape, (void *) mat_rs.data);
  if (!input_mat->GetData())
  {
#ifdef LITETNN_DEBUG
    std::cout << "input_mat == nullptr! transform failed\n";
#endif
  }
}

void TNNSSRNet::detect(const cv::Mat &mat, types::Age &age)
{
  if (mat.empty()) return;

  // 1. make input mat
  this->transform(mat);
  // 2. set input_mat
  tnn::MatConvertParam input_cvt_param;
  input_cvt_param.scale = scale_vals;
  input_cvt_param.bias = bias_vals;

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

  // 4. fetch.
  tnn::MatConvertParam cvt_param;
  std::shared_ptr<tnn::Mat> age_mat; // (1,1)
  status = instance->GetOutputMat(age_mat, cvt_param, "age", output_device_type);

  if (status != tnn::TNN_OK)
  {
#ifdef LITETNN_DEBUG
    std::cout << status.description().c_str() << "\n";
#endif
    return;
  }

  const float *age_ptr = (float *) age_mat->GetData();
  const float pred_age = age_ptr[0];

  const unsigned int interval_min = static_cast<int>(pred_age - 2.f > 0.f ? pred_age - 2.f : 0.f);
  const unsigned int interval_max = static_cast<int>(pred_age + 3.f < 100.f ? pred_age + 3.f : 100.f);

  age.age = pred_age;
  age.age_interval[0] = interval_min;
  age.age_interval[1] = interval_max;
  age.interval_prob = 1.0f;
  age.flag = true;
}























