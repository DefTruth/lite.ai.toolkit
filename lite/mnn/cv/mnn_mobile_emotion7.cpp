//
// Created by DefTruth on 2021/11/27.
//

#include "mnn_mobile_emotion7.h"

using mnncv::MNNMobileEmotion7;

MNNMobileEmotion7::MNNMobileEmotion7(const std::string &_mnn_path, unsigned int _num_threads)
    : BasicMNNHandler(_mnn_path, _num_threads)
{
  initialize_pretreat();
}

inline void MNNMobileEmotion7::initialize_pretreat()
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

void MNNMobileEmotion7::transform(const cv::Mat &mat)
{
  cv::Mat canvas;
  cv::resize(mat, canvas, cv::Size(input_width, input_height));
  // (1,3,224,224)
  pretreat->convert(canvas.data, input_width, input_height, canvas.step[0], input_tensor);
}

void MNNMobileEmotion7::detect(const cv::Mat &mat, types::Emotions &emotions)
{
  if (mat.empty()) return;
  // 1. make input tensor
  this->transform(mat);
  // 2. inference
  mnn_interpreter->runSession(mnn_session);
  auto output_tensors = mnn_interpreter->getSessionOutputAll(mnn_session);
  // 3. fetch.
  auto device_emotion_probs_ptr = output_tensors.at("emotion_preds"); // (1,7)
  MNN::Tensor host_emotion_probs_tensor(device_emotion_probs_ptr, device_emotion_probs_ptr->getDimensionType());
  device_emotion_probs_ptr->copyToHostTensor(&host_emotion_probs_tensor);

  auto emotion_dims = host_emotion_probs_tensor.shape();
  const unsigned int num_emotions = emotion_dims.at(1); // 7

  unsigned int pred_label = 0;
  const float *pred_probs_ptr = host_emotion_probs_tensor.host<float>();

  float pred_score = pred_probs_ptr[0];

  for (unsigned int i = 0; i < num_emotions; ++i)
  {
    if (pred_probs_ptr[i] > pred_score)
    {
      pred_score = pred_probs_ptr[i];
      pred_label = i;
    }
  }

  emotions.label = pred_label;
  emotions.score = pred_score;
  emotions.text = emotion_texts[pred_label];
  emotions.flag = true;
}