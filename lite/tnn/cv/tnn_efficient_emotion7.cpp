//
// Created by DefTruth on 2021/11/27.
//

#include "tnn_efficient_emotion7.h"
#include "lite/utils.h"

using tnncv::TNNEfficientEmotion7;

TNNEfficientEmotion7::TNNEfficientEmotion7(const std::string &_proto_path,
                                           const std::string &_model_path,
                                           unsigned int _num_threads) :
    BasicTNNHandler(_proto_path, _model_path, _num_threads)
{
}

void TNNEfficientEmotion7::transform(const cv::Mat &mat_rs)
{
  // push into input_mat (1,3,224,224)
  input_mat = std::make_shared<tnn::Mat>(input_device_type, tnn::N8UC3,
                                         input_shape, (void *) mat_rs.data);
  if (!input_mat->GetData())
  {
#ifdef LITETNN_DEBUG
    std::cout << "input_mat == nullptr! transform failed\n";
#endif
  }
}

void TNNEfficientEmotion7::detect(const cv::Mat &mat, types::Emotions &emotions)
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
  std::shared_ptr<tnn::Mat> emotion_logits; // (1,7)
  status = instance->GetOutputMat(emotion_logits, cvt_param, "logits", output_device_type);

  if (status != tnn::TNN_OK)
  {
#ifdef LITETNN_DEBUG
    std::cout << status.description().c_str() << "\n";
#endif
    return;
  }

  auto emotion_dims = emotion_logits->GetDims();
  const unsigned int num_emotions = emotion_dims.at(1); // 7

  unsigned int pred_label = 0;
  const float *pred_logits_ptr = (float *) emotion_logits->GetData();

  auto softmax_probs = lite::utils::math::softmax<float>(pred_logits_ptr, num_emotions, pred_label);
  emotions.label = pred_label;
  emotions.score = softmax_probs[pred_label];
  emotions.text = emotion_texts[pred_label];
  emotions.flag = true;
}
