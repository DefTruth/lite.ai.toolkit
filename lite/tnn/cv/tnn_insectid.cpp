//
// Created by DefTruth on 2022/3/27.
//

#include "tnn_insectid.h"
#include "lite/utils.h"

using tnncv::TNNInsectID;

TNNInsectID::TNNInsectID(const std::string &_proto_path,
                     const std::string &_model_path,
                     unsigned int _num_threads) :
    BasicTNNHandler(_proto_path, _model_path, _num_threads)
{
}

void TNNInsectID::transform(const cv::Mat &mat_rs)
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

void TNNInsectID::detect(const cv::Mat &mat, types::ImageNetContent &content, unsigned int top_k)
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
  std::shared_ptr<tnn::Mat> logits_mat; // (1,2037)
  status = instance->GetOutputMat(logits_mat, cvt_param, "477", output_device_type);

  if (status != tnn::TNN_OK)
  {
#ifdef LITETNN_DEBUG
    std::cout << status.description().c_str() << "\n";
#endif
    return;
  }

  auto logits_dims = logits_mat->GetDims();
  const unsigned int num_classes = logits_dims.at(1); // 2037
  const float *logits = (float *) logits_mat->GetData();

  unsigned int max_id;
  std::vector<float> scores = lite::utils::math::softmax<float>(logits, num_classes, max_id);
  std::vector<unsigned int> sorted_indices = lite::utils::math::argsort<float>(scores);
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