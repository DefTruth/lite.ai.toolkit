//
// Created by DefTruth on 2021/7/18.
//

#include "pose_robust_face.h"
#include "ort/core/ort_utils.h"

using ortcv::PoseRobustFace;


PoseRobustFace::PoseRobustFace(const std::string &_onnx_path, unsigned int _num_threads) :
    log_id(_onnx_path.data()), num_threads(_num_threads)
{
#ifdef LITE_WIN32
  std::wstring _w_onnx_path(ortcv::utils::to_wstring(_onnx_path));
  onnx_path = _w_onnx_path.data();
#else
  onnx_path = _onnx_path.data();
#endif
  ort_env = Ort::Env(ORT_LOGGING_LEVEL_ERROR, log_id);
  // 0. session options
  Ort::SessionOptions session_options;
  session_options.SetIntraOpNumThreads(num_threads);
  session_options.SetGraphOptimizationLevel(
      GraphOptimizationLevel::ORT_ENABLE_EXTENDED);
  session_options.SetLogSeverityLevel(4);
  // 1. session
  ort_session = new Ort::Session(ort_env, onnx_path, session_options);

  Ort::AllocatorWithDefaultOptions allocator;
  // 2. input name & input dims
  num_inputs = ort_session->GetInputCount();
  input_node_names.resize(num_inputs);
  // 3. initial input node dims. "input" & "yaw"
  for (unsigned int i = 0; i < num_inputs; ++i)
  {
    input_node_names[i] = ort_session->GetInputName(i, allocator);
    Ort::TypeInfo type_info = ort_session->GetInputTypeInfo(i);
    auto tensor_info = type_info.GetTensorTypeAndShapeInfo();
    unsigned int input_size = 1;
    auto input_dims = tensor_info.GetShape();
    for (unsigned int j = 0; j < input_dims.size(); ++j)
      input_size *= input_dims.at(j);
    input_tensor_sizes.push_back(input_size);
    input_node_dims.push_back(input_dims);
  }
  input_values_handler.resize(input_tensor_sizes.at(0)); // 1x3x224x224
  yaw_values_handler.resize(input_tensor_sizes.at(1)); // 1
  // 4. output names & output dims
  num_outputs = ort_session->GetOutputCount();
  output_node_names.resize(num_outputs);
  for (unsigned int i = 0; i < num_outputs; ++i)
  {
    output_node_names[i] = ort_session->GetOutputName(i, allocator);
    Ort::TypeInfo output_type_info = ort_session->GetOutputTypeInfo(i);
    auto output_tensor_info = output_type_info.GetTensorTypeAndShapeInfo();
    auto output_dims = output_tensor_info.GetShape();
    output_node_dims.push_back(output_dims);
  }
#if LITEORT_DEBUG
  this->print_debug_string();
#endif
}

PoseRobustFace::~PoseRobustFace()
{
  if (ort_session)
    delete ort_session;
  ort_session = nullptr;
}

void PoseRobustFace::print_debug_string()
{
  std::cout << "LITEORT_DEBUG LogId: " << onnx_path << "\n";
  std::cout << "=============== Inputs ==============\n";
  for (unsigned int i = 0; i < num_inputs; ++i)
    for (unsigned int j = 0; j < input_node_dims.at(i).size(); ++j)
      std::cout << "Input: " << i << " Name: "
                << input_node_names.at(i) << " Dim: " << j << " :"
                << input_node_dims.at(i).at(j) << std::endl;
  std::cout << "=============== Outputs ==============\n";
  for (unsigned int i = 0; i < num_outputs; ++i)
    for (unsigned int j = 0; j < output_node_dims.at(i).size(); ++j)
      std::cout << "Output: " << i << " Name: "
                << input_node_names.at(i) << " Dim: " << j << " :"
                << input_node_dims.at(i).at(j) << std::endl;
}


Ort::Value PoseRobustFace::transform(const cv::Mat &mat)
{
  cv::Mat canva = mat.clone();
  const unsigned int height = input_node_dims.at(0).at(2);
  const unsigned int width = input_node_dims.at(0).at(3);
  cv::resize(canva, canva, cv::Size(width, height));
  cv::cvtColor(canva, canva, cv::COLOR_BGR2RGB);
  // (1,3,224,224)
  ortcv::utils::transform::normalize_inplace(canva, mean_val, scale_val); // (0.,1.)
  return ortcv::utils::transform::create_tensor(
      canva, input_node_dims.at(0), memory_info_handler,
      input_values_handler, ortcv::utils::transform::CHW);
}

void PoseRobustFace::detect(const cv::Mat &mat, types::FaceContent &face_content, float yaw)
{
  if (mat.empty()) return;
  // 1. make input tensor
  std::vector<Ort::Value> input_tensors;
  yaw_values_handler[0] = yaw;
  input_tensors.emplace_back(this->transform(mat)); // input x (1,3,224,224)
  input_tensors.emplace_back(Ort::Value::CreateTensor<float>(
      memory_info_handler, yaw_values_handler.data(),
      input_tensor_sizes.at(1),
      input_node_dims.at(1).data(),
      input_node_dims.at(1).size()
  )); // yaw (1,)
  // 2. inference
  auto output_tensors = ort_session->Run(
      Ort::RunOptions{nullptr}, input_node_names.data(),
      input_tensors.data(), num_inputs, output_node_names.data(),
      num_outputs
  );
  Ort::Value &embedding = output_tensors.at(0);
  auto embedding_dims = output_node_dims.at(0); // (1,256)
  const unsigned int hidden_dim = embedding_dims.at(1); // 256
  const float *embedding_values = embedding.GetTensorMutableData<float>();
  std::vector<float> embedding_norm(embedding_values, embedding_values + hidden_dim);
  cv::normalize(embedding_norm, embedding_norm); // l2 normalize
  face_content.embedding.assign(embedding_norm.begin(), embedding_norm.end());
  face_content.dim = hidden_dim;
  face_content.flag = true;
}