//
// Created by YanJun Qiu on 2021/3/30.
//
#include "ort_handler.h"

using ortcore::BasicOrtHandler;

BasicOrtHandler::BasicOrtHandler(
    const std::string &_onnx_path, unsigned int _num_threads) :
    onnx_path(_onnx_path.data()), num_threads(_num_threads)
    { initialize_handler(); }

void BasicOrtHandler::initialize_handler() {
  ort_env = ort::Env(ORT_LOGGING_LEVEL_ERROR, onnx_path);
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
  input_values_handler.resize(input_tensor_size);
  // 4. output names & output dimms
  num_outputs = ort_session->GetOutputCount();
  output_node_names.resize(num_outputs);
  for (unsigned int i = 0; i < num_outputs; ++i) {
    output_node_names[i] = ort_session->GetOutputName(i, allocator);
    ort::TypeInfo output_type_info = ort_session->GetOutputTypeInfo(i);
    auto output_tensor_info = output_type_info.GetTensorTypeAndShapeInfo();
    auto output_dims = output_tensor_info.GetShape();
    output_node_dims.push_back(output_dims);
  }
#if LITEORT_DEBUG
  this->print_debug_string();
#endif
}

BasicOrtHandler::~BasicOrtHandler() {
  if (ort_session)
    delete ort_session;
  ort_session = nullptr;
}

void BasicOrtHandler::print_debug_string() {
  std::cout << "LITEORT_DEBUG LogId: " << onnx_path << "\n";
  std::cout << "=============== Input-Dims ==============\n";
  for (unsigned int i = 0; i < input_node_dims.size(); ++i)
    std::cout << "input_node_dims: " << input_node_dims.at(i) << "\n";
  std::cout << "=============== Output-Dims ==============\n";
  for (unsigned int i = 0; i < num_outputs; ++i)
    for (unsigned int j = 0; j < output_node_dims.at(i).size(); ++j)
      std::cout << "Output: " << i << " Name: "
                << output_node_names.at(i) << " Dim: " << j << " :"
                << output_node_dims.at(i).at(j) << std::endl;
}