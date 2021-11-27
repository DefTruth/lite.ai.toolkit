//
// Created by DefTruth on 2021/11/27.
//

#include "tnn_gender_googlenet.h"
#include "lite/utils.h"

using tnncv::TNNGenderGoogleNet;

TNNGenderGoogleNet::TNNGenderGoogleNet(const std::string &_proto_path,
                                       const std::string &_model_path,
                                       unsigned int _num_threads) :
    BasicTNNHandler(_proto_path, _model_path, _num_threads)
{
}

void TNNGenderGoogleNet::transform(const cv::Mat &mat)
{
  cv::Mat mat_rs;
  cv::resize(mat, mat_rs, cv::Size(input_width, input_height));
  cv::cvtColor(mat_rs, mat_rs, cv::COLOR_BGR2RGB);
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

void TNNGenderGoogleNet::detect(const cv::Mat &mat, types::Gender &gender)
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
  std::shared_ptr<tnn::Mat> gender_logits; // (1,8)
  status = instance->GetOutputMat(gender_logits, cvt_param, "loss3/loss3_Y", output_device_type);

  if (status != tnn::TNN_OK)
  {
#ifdef LITETNN_DEBUG
    std::cout << status.description().c_str() << "\n";
#endif
    return;
  }

  auto gender_dims = gender_logits->GetDims();
  const unsigned int num_genders = gender_dims.at(1); // 2
  const float *pred_logits_ptr = (float *) gender_logits->GetData();

  unsigned int pred_gender = 0;
  auto softmax_probs = lite::utils::math::softmax<float>(pred_logits_ptr, num_genders, pred_gender);
  unsigned int gender_label = pred_gender == 1 ? 0 : 1;
  gender.label = gender_label;
  gender.text = gender_texts[gender_label];
  gender.score = softmax_probs[pred_gender];
  gender.flag = true;
}