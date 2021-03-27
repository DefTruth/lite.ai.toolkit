//
// Created by YanJun Qiu on 2021/3/14.
//

#include "ultraface.h"

using ortcv::Box;
using ortcv::UltraFace;

UltraFace::UltraFace(const std::string &_onnx_path, int _input_height, int _input_width,
                     unsigned int _num_threads) : onnx_path(_onnx_path.data()),
                                                  input_height(_input_height),
                                                  input_width(_input_width),
                                                  num_threads(_num_threads) {
  ort_env = ort::Env(ORT_LOGGING_LEVEL_ERROR, "ultraface-onnx");
  // 0. session options
  ort::SessionOptions session_options;
  session_options.SetIntraOpNumThreads(num_threads);
  session_options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_EXTENDED);
  session_options.SetLogSeverityLevel(4);
  // 1. session
  ort_session = new ort::Session(ort_env, onnx_path, session_options);

  ort::AllocatorWithDefaultOptions allocator;
  // 2. input name & input dims
  input_name = ort_session->GetInputName(0, allocator);
  input_node_names.resize(1);
  input_node_names[0] = input_name;
  // 3. type info.
  ort::TypeInfo type_info = ort_session->GetInputTypeInfo(0);
  auto tensor_info = type_info.GetTensorTypeAndShapeInfo();
  input_tensor_size = 1;
  input_node_dims = tensor_info.GetShape();
  for (unsigned int i = 0; i < input_node_dims.size(); ++i)
    input_tensor_size *= input_node_dims.at(i);
#if LITEORT_DEBUG
  for (unsigned int i = 0; i < input_node_dims.size(); ++i)
    std::cout << "input_node_dims: " << input_node_dims.at(i) << "\n";
#endif
  input_tensor_values.resize(input_tensor_size);
  // 4. output names & output dimms
  num_outputs = ort_session->GetOutputCount();
#if LITEORT_DEBUG
  std::cout << "num_outputs: " << num_outputs << "\n";
  assert(num_outputs > 0 ? 1 : 0);
#endif
  output_node_names.resize(num_outputs);
  for (unsigned int i = 0; i < num_outputs; ++i) {
    output_node_names[i] = ort_session->GetOutputName(i, allocator);
    ort::TypeInfo output_type_info = ort_session->GetOutputTypeInfo(i);
    auto output_tensor_info = output_type_info.GetTensorTypeAndShapeInfo();
    auto output_dims = output_tensor_info.GetShape();
    output_node_dims.push_back(output_dims);
  }
#if LITEORT_DEBUG
  std::cout << "=============== Output-Dims ==============\n";
  for (unsigned int i = 0; i < num_outputs; ++i)
    for (unsigned int j = 0; j < output_node_dims.at(i).size(); ++j)
      std::cout << "Output: " << i << " Name: "
                << output_node_names.at(i) << " Dim: "
                << output_node_dims.at(i).at(j) << std::endl;
#endif
}

UltraFace::~UltraFace() {
  if (ort_session)
    delete ort_session;
  ort_session = nullptr;
}

void UltraFace::preprocess(const cv::Mat &mat) {
  cv::Mat canva = mat.clone();
  // 0.BGR -> RGB assume the channel order of input img is BGR.
  cv::cvtColor(canva, canva, cv::COLOR_BGR2RGB);
  // 1. resize & normalize
  cv::Mat resize_norm;
  cv::resize(canva, canva, cv::Size(input_width, input_height));
  canva.convertTo(resize_norm, CV_32FC3); // first, convert to float32.
  resize_norm = (resize_norm - mean_val) / scale_val; // then, normalize.

  std::vector<cv::Mat> channels;
  cv::split(resize_norm, channels);
  std::vector<float> channel_values;
  channel_values.resize(input_height * input_width);
  for (int i = 0; i < channels.size(); ++i) {
    channel_values.clear();
    channel_values = channels.at(i).reshape(1, 1); // flatten
    std::memcpy(input_tensor_values.data() + i * (input_height * input_width),
                channel_values.data(),
                input_height * input_width * sizeof(float)); // CXHXW
  }
}

void UltraFace::detect(const cv::Mat &mat, std::vector<Box> &detected_boxes,
                       float score_threshold, float iou_threshold, int top_k) {
  this->preprocess(mat);


}

void UltraFace::generate_bounding_boxes(std::vector<Box> &bbox_collection,
                                        const float *scores, const float *boxes,
                                        float score_threshold, int num_anchors) {
}

void UltraFace::nms(std::vector<Box> &input, std::vector<Box> &output, int type) {

}

cv::Mat UltraFace::draw_boxes(const cv::Mat &mat, const std::vector<Box> &_boxes) {
  cv::Mat canva;
  return canva;
}

void UltraFace::draw_boxes_inplane(cv::Mat &mat_inplane, const std::vector<Box> &_boxes) {

}