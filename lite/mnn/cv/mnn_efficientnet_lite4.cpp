//
// Created by DefTruth on 2021/11/29.
//

#include "mnn_efficientnet_lite4.h"
#include "lite/utils.h"

using mnncv::MNNEfficientNetLite4;

MNNEfficientNetLite4::MNNEfficientNetLite4(const std::string &_mnn_path, unsigned int _num_threads)
    : BasicMNNHandler(_mnn_path, _num_threads)
{
  // re-init with fixed input shape, due to the error of input
  // shape auto-detection while using MNN with NHWC input.
  // TODO: pre-process bug fix
  dimension_type = MNN::Tensor::TENSORFLOW;
  input_batch = 1;
  input_channel = 3;
  input_width = 224;
  input_height = 224;
  mnn_interpreter->resizeTensor(
      input_tensor, {input_batch, input_height, input_width, input_channel});
  mnn_interpreter->resizeSession(mnn_session);

  initialize_pretreat();
}

inline void MNNEfficientNetLite4::initialize_pretreat()
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

void MNNEfficientNetLite4::transform(const cv::Mat &mat)
{
  cv::Mat canvas;
  cv::resize(mat, canvas, cv::Size(input_width, input_height));
  // (1,224,224,3)
  pretreat->convert(canvas.data, input_width, input_height, canvas.step[0], input_tensor);
}

void MNNEfficientNetLite4::detect(const cv::Mat &mat, types::ImageNetContent &content, unsigned int top_k)
{
  if (mat.empty()) return;
  // 1. make input tensor
  this->transform(mat);
  // 2. inference
  mnn_interpreter->runSession(mnn_session);
  auto output_tensors = mnn_interpreter->getSessionOutputAll(mnn_session);
  // 3. fetch.
  auto device_scores_ptr = output_tensors.at("Softmax:0");
  MNN::Tensor host_scores_tensor(device_scores_ptr, device_scores_ptr->getDimensionType());
  device_scores_ptr->copyToHostTensor(&host_scores_tensor);

  auto scores_dims = host_scores_tensor.shape();
  const unsigned int num_classes = scores_dims.at(1); // 1000
  const float *scores = host_scores_tensor.host<float>();

  std::vector<unsigned int> sorted_indices = lite::utils::math::argsort<float>(scores, num_classes);
  if (top_k > num_classes) top_k = num_classes;

  content.scores.clear();
  content.labels.clear();
  content.texts.clear();
  for (unsigned int i = 0; i < top_k; ++i)
  {
    content.labels.push_back(sorted_indices[i]);
    content.scores.push_back(scores[sorted_indices[i]]);
    content.texts.push_back(class_names[sorted_indices[i]]);
  }
  content.flag = true;
}