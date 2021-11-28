//
// Created by DefTruth on 2021/11/27.
//

#include "mnn_efficient_emotion8.h"
#include "lite/utils.h"

using mnncv::MNNEfficientEmotion8;

MNNEfficientEmotion8::MNNEfficientEmotion8(const std::string &_mnn_path, unsigned int _num_threads)
    : BasicMNNHandler(_mnn_path, _num_threads)
{
  initialize_pretreat();
}

inline void MNNEfficientEmotion8::initialize_pretreat()
{
  pretreat = std::shared_ptr<MNN::CV::ImageProcess>(
      MNN::CV::ImageProcess::create(
          MNN::CV::BGR,
          MNN::CV::RGB,
          mean_vals, 3,
          norm_vals, 3
      )
  );
}

void MNNEfficientEmotion8::transform(const cv::Mat &mat)
{
  cv::Mat canvas;
  cv::resize(mat, canvas, cv::Size(input_width, input_height));
  // (1,3,224,224)
  pretreat->convert(canvas.data, input_width, input_height, canvas.step[0], input_tensor);
}

void MNNEfficientEmotion8::detect(const cv::Mat &mat, types::Emotions &emotions)
{
  if (mat.empty()) return;
  // 1. make input tensor
  this->transform(mat);
  // 2. inference
  mnn_interpreter->runSession(mnn_session);
  auto output_tensors = mnn_interpreter->getSessionOutputAll(mnn_session);
  // 3. fetch.
  auto device_emotion_logits_ptr = output_tensors.at("logits"); // (1,8)
  MNN::Tensor host_emotion_logits_tensor(device_emotion_logits_ptr, device_emotion_logits_ptr->getDimensionType());
  device_emotion_logits_ptr->copyToHostTensor(&host_emotion_logits_tensor);

  auto emotion_dims = host_emotion_logits_tensor.shape();
  const unsigned int num_emotions = emotion_dims.at(1); // 8

  unsigned int pred_label = 0;
  const float *pred_logits_ptr = host_emotion_logits_tensor.host<float>();

  auto softmax_probs = lite::utils::math::softmax<float>(pred_logits_ptr, num_emotions, pred_label);
  emotions.label = pred_label;
  emotions.score = softmax_probs[pred_label];
  emotions.text = emotion_texts[pred_label];
  emotions.flag = true;
}