//
// Created by DefTruth on 2021/11/27.
//

#include "tnn_age_googlenet.h"
#include "lite/utils.h"

using tnncv::TNNAgeGoogleNet;

TNNAgeGoogleNet::TNNAgeGoogleNet(const std::string &_proto_path,
                                 const std::string &_model_path,
                                 unsigned int _num_threads) :
    BasicTNNHandler(_proto_path, _model_path, _num_threads)
{
}

void TNNAgeGoogleNet::transform(const cv::Mat &mat_rs)
{
  // be carefully, no deepcopy inside this tnn::Mat constructor,
  // so, we can not pass a local cv::Mat to this constructor.
  // push into input_mat
  input_mat = std::make_shared<tnn::Mat>(input_device_type, tnn::N8UC3,
                                         input_shape, (void *) mat_rs.data);
  if (!input_mat->GetData())
  {
#ifdef LITETNN_DEBUG
    std::cout << "input_mat == nullptr! transform failed\n";
#endif
  }
}

void TNNAgeGoogleNet::detect(const cv::Mat &mat, types::Age &age)
{
  if (mat.empty()) return;

  // 1. make input mat
  cv::Mat mat_rs;
  cv::resize(mat, mat_rs, cv::Size(input_width, input_height));
  cv::cvtColor(mat_rs, mat_rs, cv::COLOR_BGR2RGB);
  this->transform(mat_rs);
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
  std::shared_ptr<tnn::Mat> age_logits; // (1,8)
  status = instance->GetOutputMat(age_logits, cvt_param, "loss3/loss3_Y", output_device_type);

  if (status != tnn::TNN_OK)
  {
#ifdef LITETNN_DEBUG
    std::cout << status.description().c_str() << "\n";
#endif
    return;
  }

  auto age_dims = age_logits->GetDims();
  unsigned int interval = 0;
  const unsigned int num_intervals = age_dims.at(1); // 8
  const float *pred_logits_ptr = (float *) age_logits->GetData();

  auto softmax_probs = lite::utils::math::softmax<float>(pred_logits_ptr, num_intervals, interval);
  const float pred_age = static_cast<float>(age_intervals[interval][0] + age_intervals[interval][1]) / 2.0f;

  age.age = pred_age;
  age.age_interval[0] = age_intervals[interval][0];
  age.age_interval[1] = age_intervals[interval][1];
  age.interval_prob = softmax_probs[interval];
  age.flag = true;
}