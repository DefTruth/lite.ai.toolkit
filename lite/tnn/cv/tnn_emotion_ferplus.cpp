//
// Created by DefTruth on 2021/11/27.
//

#include "tnn_emotion_ferplus.h"
#include "lite/utils.h"

using tnncv::TNNEmotionFerPlus;

TNNEmotionFerPlus::TNNEmotionFerPlus(const std::string &_proto_path,
                                     const std::string &_model_path,
                                     unsigned int _num_threads) :
    BasicTNNHandler(_proto_path, _model_path, _num_threads)
{
}

void TNNEmotionFerPlus::transform(const cv::Mat &mat)
{
  cv::Mat mat_rs;
  cv::resize(mat, mat_rs, cv::Size(input_width, input_height));
  cv::cvtColor(mat_rs, mat_rs, cv::COLOR_BGR2GRAY);
  // push into input_mat (1,1,64,64)
  input_mat = std::make_shared<tnn::Mat>(input_device_type, tnn::NGRAY,
                                         input_shape, (void *) mat_rs.data);
  if (!input_mat->GetData())
  {
#ifdef LITETNN_DEBUG
    std::cout << "input_mat == nullptr! transform failed\n";
#endif
  }
}

void TNNEmotionFerPlus::detect(const cv::Mat &mat, types::Emotions &emotions)
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
  std::shared_ptr<tnn::Mat> emotion_logits; // (1,8)
  status = instance->GetOutputMat(emotion_logits, cvt_param, "Plus692_Output_0", output_device_type);

  if (status != tnn::TNN_OK)
  {
#ifdef LITETNN_DEBUG
    std::cout << status.description().c_str() << "\n";
#endif
    return;
  }

  auto emotion_dims = emotion_logits->GetDims();
  const unsigned int num_emotions = emotion_dims.at(1); // 8

  unsigned int pred_label = 0;
  const float *pred_logits_ptr = (float *) emotion_logits->GetData();

  auto softmax_probs = lite::utils::math::softmax<float>(pred_logits_ptr, num_emotions, pred_label);
  emotions.label = pred_label;
  emotions.score = softmax_probs[pred_label];
  emotions.text = emotion_texts[pred_label];
  emotions.flag = true;
}
