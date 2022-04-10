//
// Created by DefTruth on 2022/4/9.
//

#include "backgroundmattingv2.h"
#include "lite/ort/core/ort_utils.h"
#include "lite/utils.h"

using ortcv::BackgroundMattingV2;

BackgroundMattingV2::BackgroundMattingV2(const std::string &_onnx_path, unsigned int _num_threads) :
    log_id(_onnx_path.data()), num_threads(_num_threads)
{
#ifdef LITE_WIN32
  std::wstring _w_onnx_path(lite::utils::to_wstring(_onnx_path));
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
  // GPU Compatibility.
#ifdef USE_CUDA
  OrtSessionOptionsAppendExecutionProvider_CUDA(session_options, 0); // C API stable.
#endif
  ort_session = new Ort::Session(ort_env, onnx_path, session_options);

  // 3. type info.
  num_inputs = ort_session->GetInputCount(); // 2
  input_node_names.resize(num_inputs);
  for (unsigned int i = 0; i < num_inputs; ++i)
    input_node_names[i] = ort_session->GetInputName(i, allocator);

  Ort::TypeInfo input_mat_type_info = ort_session->GetInputTypeInfo(0);
  Ort::TypeInfo input_bgr_type_info = ort_session->GetInputTypeInfo(1);
  auto input_mat_tensor_info = input_mat_type_info.GetTensorTypeAndShapeInfo();
  auto input_bgr_tensor_info = input_bgr_type_info.GetTensorTypeAndShapeInfo();
  auto input_mat_shape = input_mat_tensor_info.GetShape();
  auto input_bgr_shape = input_bgr_tensor_info.GetShape();
  input_node_dims.push_back(input_mat_shape);
  input_node_dims.push_back(input_bgr_shape);
  unsigned int input_mat_tensor_size = 1;
  unsigned int input_bgr_tensor_size = 1;
  for (unsigned int i = 0; i < input_mat_shape.size(); ++i)
    input_mat_tensor_size *= input_mat_shape.at(i);
  for (unsigned int i = 0; i < input_bgr_shape.size(); ++i)
    input_bgr_tensor_size *= input_bgr_shape.at(i);
  input_mat_value_handler.resize(input_mat_tensor_size);
  input_bgr_value_handler.resize(input_bgr_tensor_size);

  num_outputs = ort_session->GetOutputCount();
  output_node_names.resize(num_outputs);
  for (unsigned int i = 0; i < num_outputs; ++i)
  {
    output_node_names[i] = ort_session->GetOutputName(i, allocator);
    Ort::TypeInfo type_info = ort_session->GetOutputTypeInfo(i);
    auto tensor_info = type_info.GetTensorTypeAndShapeInfo();
    auto output_shape = tensor_info.GetShape();
    output_node_dims.push_back(output_shape);
  }

#if LITEORT_DEBUG
  std::cout << "Load " << onnx_path << " done!" << std::endl;
  this->print_debug_string();
#endif
}

BackgroundMattingV2::~BackgroundMattingV2()
{
  if (ort_session)
    delete ort_session;
  ort_session = nullptr;
}

void BackgroundMattingV2::print_debug_string()
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
                << output_node_names.at(i) << " Dim: " << j << " :"
                << output_node_dims.at(i).at(j) << std::endl;
}

std::vector<Ort::Value> BackgroundMattingV2::transform(const cv::Mat &mat, const cv::Mat &bgr)
{
  cv::Mat mat_rs, bgr_rs;
  cv::resize(mat, mat_rs, cv::Size(input_node_dims.at(0).at(3), input_node_dims.at(0).at(2)));
  cv::resize(bgr, bgr_rs, cv::Size(input_node_dims.at(1).at(3), input_node_dims.at(1).at(2)));
  cv::cvtColor(mat_rs, mat_rs, cv::COLOR_BGR2RGB);
  cv::cvtColor(bgr_rs, bgr_rs, cv::COLOR_BGR2RGB);

  ortcv::utils::transform::normalize_inplace(mat_rs, mean_val, scale_val); // float32
  ortcv::utils::transform::normalize_inplace(bgr_rs, mean_val, scale_val); // float32
  // e.g (1,3,512,512)
  std::vector<Ort::Value> input_tensors;
  input_tensors.emplace_back(ortcv::utils::transform::create_tensor(
      mat_rs, input_node_dims.at(0), memory_info_handler,
      input_mat_value_handler, ortcv::utils::transform::CHW
  )); // deepcopy inside
  input_tensors.emplace_back(ortcv::utils::transform::create_tensor(
      bgr_rs, input_node_dims.at(1), memory_info_handler,
      input_bgr_value_handler, ortcv::utils::transform::CHW
  )); // deepcopy inside
  return input_tensors;
}


void BackgroundMattingV2::detect(const cv::Mat &mat, const cv::Mat &bgr,
                                 types::MattingContent &content, bool remove_noise,
                                 bool minimum_post_process)
{
  if (mat.empty() || bgr.empty()) return;
  // 1. make input tensor
  auto input_tensors = this->transform(mat, bgr);
  // 2. inference
  auto output_tensors = ort_session->Run(
      Ort::RunOptions{nullptr}, input_node_names.data(),
      input_tensors.data(), num_inputs, output_node_names.data(), num_outputs
  );
  // 3. generate matting
  this->generate_matting(output_tensors, mat, content, remove_noise, minimum_post_process);
}

void BackgroundMattingV2::generate_matting(std::vector<Ort::Value> &output_tensors,
                                           const cv::Mat &mat, types::MattingContent &content,
                                           bool remove_noise, bool minimum_post_process)
{
  Ort::Value &pha = output_tensors.at(0); // e.g (1,1,512,512)
  Ort::Value &fgr = output_tensors.at(1); // e.g (1,3,512,512)
  const unsigned int h = mat.rows;
  const unsigned int w = mat.cols;

  auto output_dims = pha.GetTensorTypeAndShapeInfo().GetShape();
  const unsigned int out_h = output_dims.at(2);
  const unsigned int out_w = output_dims.at(3);
  const unsigned int channel_step = out_h * out_w;

  float *pha_ptr = pha.GetTensorMutableData<float>();
  float *fgr_ptr = fgr.GetTensorMutableData<float>();
  // fast assign & channel transpose(CHW->HWC).
  cv::Mat pmat(out_h, out_w, CV_32FC1, pha_ptr);
  if (remove_noise) lite::utils::remove_small_connected_area(pmat, 0.05f);

  std::vector<cv::Mat> fgr_channel_mats;
  cv::Mat rmat(out_h, out_w, CV_32FC1, fgr_ptr);
  cv::Mat gmat(out_h, out_w, CV_32FC1, fgr_ptr + channel_step);
  cv::Mat bmat(out_h, out_w, CV_32FC1, fgr_ptr + 2 * channel_step);
  rmat *= 255.;
  bmat *= 255.;
  gmat *= 255.;
  fgr_channel_mats.push_back(bmat);
  fgr_channel_mats.push_back(gmat);
  fgr_channel_mats.push_back(rmat);

  content.pha_mat = pmat;
  cv::merge(fgr_channel_mats, content.fgr_mat);
  content.fgr_mat.convertTo(content.fgr_mat, CV_8UC3);

  if (!minimum_post_process)
  {
    std::vector<cv::Mat> merge_channel_mats;
    cv::Mat rest = 1. - pmat;
    cv::Mat mbmat = bmat.mul(pmat) + rest * 153.;
    cv::Mat mgmat = gmat.mul(pmat) + rest * 255.;
    cv::Mat mrmat = rmat.mul(pmat) + rest * 120.;
    merge_channel_mats.push_back(mbmat);
    merge_channel_mats.push_back(mgmat);
    merge_channel_mats.push_back(mrmat);
    cv::merge(merge_channel_mats, content.merge_mat);
    content.merge_mat.convertTo(content.merge_mat, CV_8UC3);
  }

  // resize alpha
  if (out_h != h || out_w != w)
  {
    cv::resize(content.pha_mat, content.pha_mat, cv::Size(w, h));
    cv::resize(content.fgr_mat, content.fgr_mat, cv::Size(w, h));
    if (!minimum_post_process)
      cv::resize(content.merge_mat, content.merge_mat, cv::Size(w, h));
  }

  content.flag = true;
}
































