//
// Created by DefTruth on 2022/6/3.
//
#include "head_seg.h"
#include "lite/ort/core/ort_utils.h"

#ifdef LITE_WIN32
#include "lite/utils.h"
#endif

using ortcv::HeadSeg;

HeadSeg::HeadSeg(const std::string &_onnx_path, unsigned int _num_threads) :
    log_id(_onnx_path.data()), num_threads(_num_threads)
{
#ifdef LITE_WIN32
  std::wstring _w_onnx_path(lite::utils::to_wstring(_onnx_path));
  onnx_path = _w_onnx_path.data();
#else
  onnx_path = _onnx_path.data();
#endif
  initialize_handler();
}

HeadSeg::~HeadSeg()
{
  if (ort_session)
    delete ort_session;
  ort_session = nullptr;
}

void HeadSeg::initialize_handler()
{
  ort_env = Ort::Env(ORT_LOGGING_LEVEL_ERROR, log_id);
  // 0. session options
  Ort::SessionOptions session_options;
  session_options.SetIntraOpNumThreads(num_threads);
  session_options.SetGraphOptimizationLevel(
      GraphOptimizationLevel::ORT_ENABLE_ALL);
  session_options.SetLogSeverityLevel(4);

  // OrtCUDAProviderOptions provider_options;
  // session_options.AppendExecutionProvider_CUDA(provider_options);
#ifdef USE_CUDA
  OrtSessionOptionsAppendExecutionProvider_CUDA(session_options, 0); // C API stable.
#endif
  // 1. session
  ort_session = new Ort::Session(ort_env, onnx_path, session_options);

  Ort::AllocatorWithDefaultOptions allocator;
  // 2. input name & input dims
  input_name = ort_session->GetInputName(0, allocator);
  input_node_names.resize(1);
  input_node_names[0] = input_name;
  // 3. type info.
  Ort::TypeInfo type_info = ort_session->GetInputTypeInfo(0);
  auto tensor_info = type_info.GetTensorTypeAndShapeInfo();
  input_tensor_size = 1;
  input_node_dims = tensor_info.GetShape();
  // hard code batch size as 1
  input_node_dims.at(0) = 1;
  // 4. output names & output dimms
  num_outputs = ort_session->GetOutputCount();
  output_node_names.resize(num_outputs);
  for (unsigned int i = 0; i < num_outputs; ++i)
    output_node_names[i] = ort_session->GetOutputName(i, allocator);
#if LITEORT_DEBUG
  this->print_debug_string();
#endif
}

void HeadSeg::print_debug_string()
{
  std::cout << "LITEORT_DEBUG LogId: " << onnx_path << "\n";
  std::cout << "=============== Input-Dims ==============\n";
  for (unsigned int i = 0; i < input_node_dims.size(); ++i)
    std::cout << "input_node_dims: " << input_node_dims.at(i) << "\n";
  std::cout << "=============== Output-Dims ==============\n";
  for (unsigned int i = 0; i < num_outputs; ++i)
    std::cout << "Output: " << i << " Name: "
              << output_node_names.at(i) << std::endl;
  std::cout << "========================================\n";
}

Ort::Value HeadSeg::transform(const cv::Mat &mat_rs)
{
  cv::Mat canvas;
  cv::cvtColor(mat_rs, canvas, cv::COLOR_BGR2RGB);
  canvas.convertTo(canvas, CV_32FC3, 1.f / 255.f, 0.f);
  // (1,384,384,3) HWC
  return ortcv::utils::transform::create_tensor(
      canvas, input_node_dims, memory_info_handler,
      input_values_handler, ortcv::utils::transform::HWC
  );
}

void HeadSeg::detect(const cv::Mat &mat, types::HeadSegContent &content)
{
  if (mat.empty()) return;
  const unsigned int img_h = mat.rows;
  const unsigned int img_w = mat.cols;
  const unsigned int channels = mat.channels();
  if (channels != 3) return;
  const unsigned int input_h = input_node_dims.at(1); // 384
  const unsigned int input_w = input_node_dims.at(2); // 384

  cv::Mat mat_rs;
  cv::resize(mat, mat_rs, cv::Size(input_w, input_h));
  // 1. make input tensor
  Ort::Value input_tensor = this->transform(mat_rs);
  // 2. inference mask (1,384,384,1)
  auto output_tensors = ort_session->Run(
      Ort::RunOptions{nullptr}, input_node_names.data(),
      &input_tensor, 1, output_node_names.data(),
      num_outputs
  );
  // 3. post process.
  Ort::Value &mask_pred = output_tensors.at(0); // (1,384,384,1)
  auto mask_dims = mask_pred.GetTensorTypeAndShapeInfo().GetShape();
  const unsigned int out_h = mask_dims.at(1);
  const unsigned int out_w = mask_dims.at(2);
  float *mask_ptr = mask_pred.GetTensorMutableData<float>();

  cv::Mat mask_adj;
  cv::Mat mask_out(out_h, out_w, CV_32FC1, mask_ptr); // ref only
  cv::resize(mask_out, mask_adj, cv::Size(img_w, img_h)); // (img_h,img_w,1) allocated

  content.mask = mask_adj;
  content.flag = true;
}

