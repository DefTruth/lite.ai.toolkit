//
// Created by DefTruth on 2022/4/9.
//

#include "backgroundmattingv2_dyn.h"
#include "lite/ort/core/ort_utils.h"
#include "lite/utils.h"

using ortcv::BackgroundMattingV2Dyn;

BackgroundMattingV2Dyn::BackgroundMattingV2Dyn(const std::string &_onnx_path, unsigned int _num_threads) :
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
  // 2. input name & input dims
  input_node_names.resize(num_inputs); // num_inputs=1
  input_node_names.resize(num_inputs);
  for (unsigned int i = 0; i < num_inputs; ++i)
    input_node_names[i] = ort_session->GetInputName(i, allocator);
  // 3. initial input node dims.
  dynamic_input_node_dims.push_back({1, 3, dynamic_input_height, dynamic_input_width}); // src
  dynamic_input_node_dims.push_back({1, 3, dynamic_input_height, dynamic_input_width}); // bgr
  dynamic_input_mat_size = 1 * 3 * dynamic_input_height * dynamic_input_width;
  dynamic_input_bgr_size = 1 * 3 * dynamic_input_height * dynamic_input_width;
  dynamic_input_mat_value_handler.resize(dynamic_input_mat_size);
  dynamic_input_bgr_value_handler.resize(dynamic_input_bgr_size);
  // 4. output names & output dims
  num_outputs = ort_session->GetOutputCount();
  output_node_names.resize(num_outputs);
  for (unsigned int i = 0; i < num_outputs; ++i)
    output_node_names[i] = ort_session->GetOutputName(i, allocator);
#if LITEORT_DEBUG
  this->print_debug_string();
#endif
}

BackgroundMattingV2Dyn::~BackgroundMattingV2Dyn()
{
  if (ort_session)
    delete ort_session;
  ort_session = nullptr;
}

void BackgroundMattingV2Dyn::print_debug_string()
{
  std::cout << "LITEORT_DEBUG LogId: " << onnx_path << "\n";
  std::cout << "=============== Inputs ==============\n";
  for (unsigned int i = 0; i < num_inputs; ++i)
    for (unsigned int j = 0; j < dynamic_input_node_dims.at(i).size(); ++j)
      std::cout << "Dynamic Input: " << i << " Name: "
                << input_node_names.at(i) << " Init Dim: " << j << " :"
                << dynamic_input_node_dims.at(i).at(j) << std::endl;
  std::cout << "=============== Outputs ==============\n";
  for (unsigned int i = 0; i < num_outputs; ++i)
    std::cout << "Dynamic Output " << i << ": " << output_node_names[i] << std::endl;
}

std::vector<Ort::Value> BackgroundMattingV2Dyn::transform(const cv::Mat &mat, const cv::Mat &bgr)
{
  auto padded_mat = this->padding(mat);
  auto padded_bgr = this->padding(bgr);
  cv::cvtColor(padded_mat, padded_mat, cv::COLOR_BGR2RGB);
  cv::cvtColor(padded_bgr, padded_bgr, cv::COLOR_BGR2RGB);

  ortcv::utils::transform::normalize_inplace(padded_mat, mean_val, scale_val); // float32
  ortcv::utils::transform::normalize_inplace(padded_bgr, mean_val, scale_val); // float32
  // e.g (1,3,512,512)
  std::vector<Ort::Value> input_tensors;
  input_tensors.emplace_back(ortcv::utils::transform::create_tensor(
      padded_mat, dynamic_input_node_dims.at(0), memory_info_handler,
      dynamic_input_mat_value_handler, ortcv::utils::transform::CHW
  )); // deepcopy inside
  input_tensors.emplace_back(ortcv::utils::transform::create_tensor(
      padded_bgr, dynamic_input_node_dims.at(1), memory_info_handler,
      dynamic_input_bgr_value_handler, ortcv::utils::transform::CHW
  )); // deepcopy inside
  return input_tensors;
}

void BackgroundMattingV2Dyn::detect(const cv::Mat &mat, const cv::Mat &bgr,
                                    types::MattingContent &content, bool remove_noise,
                                    bool minimum_post_process)
{
  if (mat.empty() || bgr.empty()) return;
  const unsigned int img_height = mat.rows;
  const unsigned int img_width = mat.cols;
  const unsigned int bgr_height = mat.rows;
  const unsigned int bgr_width = mat.cols;
  cv::Mat bgr_ref;
  if (bgr_height != img_height || bgr_width != img_width)
    cv::resize(bgr, bgr_ref, cv::Size(img_width, img_height));
  else bgr_ref = bgr;

  // align input size with 32 first
  this->update_dynamic_shape(img_height, img_width);
  // 1. make input tensor
  auto input_tensors = this->transform(mat, bgr_ref);
  // 2. inference
  auto output_tensors = ort_session->Run(
      Ort::RunOptions{nullptr}, input_node_names.data(),
      input_tensors.data(), num_inputs, output_node_names.data(), num_outputs
  );
  // 3. generate matting
  this->generate_matting(output_tensors, mat, content, remove_noise, minimum_post_process);
}

void BackgroundMattingV2Dyn::generate_matting(std::vector<Ort::Value> &output_tensors,
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
  cv::Mat alpha_pred(out_h, out_w, CV_32FC1, pha_ptr); // ref only, zero copy
  cv::Mat pmat = alpha_pred(cv::Rect(align_val, align_val, w, h)); // ref only, zero copy
  if (remove_noise) lite::utils::remove_small_connected_area(pmat, 0.05f);

  std::vector<cv::Mat> fgr_channel_mats;
  cv::Mat rmat_pred(out_h, out_w, CV_32FC1, fgr_ptr); // ref only, zero copy
  cv::Mat gmat_pred(out_h, out_w, CV_32FC1, fgr_ptr + channel_step);
  cv::Mat bmat_pred(out_h, out_w, CV_32FC1, fgr_ptr + 2 * channel_step);
  cv::Mat rmat = rmat_pred(cv::Rect(align_val, align_val, w, h)); // ref only, zero copy
  cv::Mat gmat = gmat_pred(cv::Rect(align_val, align_val, w, h));
  cv::Mat bmat = bmat_pred(cv::Rect(align_val, align_val, w, h));
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

  content.flag = true;
}

void BackgroundMattingV2Dyn::update_dynamic_shape(unsigned int img_height, unsigned int img_width)
{
  // align input shape with 32
  const unsigned int h = img_height;
  const unsigned int w = img_width;
  // update dynamic input dims
  if (h % align_val == 0 && w % align_val == 0)
  {
    // aligned
    dynamic_input_height = h + 2 * align_val;
    dynamic_input_width = w + 2 * align_val;
  } // un-aligned
  else
  {
    // align first
    const unsigned int align_h = align_val * ((h - 1) / align_val + 1);
    const unsigned int align_w = align_val * ((w - 1) / align_val + 1);
    const unsigned int pad_h = align_h - h; // >= 0
    const unsigned int pad_w = align_w - w; // >= 0
    dynamic_input_height = h + align_val + (pad_h + align_val);
    dynamic_input_width = w + align_val + (pad_w + align_val);
  }
  dynamic_input_node_dims.at(0).at(2) = dynamic_input_height;
  dynamic_input_node_dims.at(0).at(3) = dynamic_input_width;
  dynamic_input_node_dims.at(1).at(2) = dynamic_input_height;
  dynamic_input_node_dims.at(1).at(3) = dynamic_input_width;
  dynamic_input_mat_size = 1 * 3 * dynamic_input_height * dynamic_input_width;
  dynamic_input_bgr_size = 1 * 3 * dynamic_input_height * dynamic_input_width;
  dynamic_input_mat_value_handler.resize(dynamic_input_mat_size);
  dynamic_input_bgr_value_handler.resize(dynamic_input_bgr_size);
}

cv::Mat BackgroundMattingV2Dyn::padding(const cv::Mat &unpad_mat)
{
  const unsigned int h = unpad_mat.rows;
  const unsigned int w = unpad_mat.cols;

  // aligned
  if (h % align_val == 0 && w % align_val == 0)
  {
    const unsigned int target_h = h + 2 * align_val;
    const unsigned int target_w = w + 2 * align_val;
    cv::Mat pad_mat(target_h, target_w, unpad_mat.type());

    cv::copyMakeBorder(unpad_mat, pad_mat, align_val, align_val,
                       align_val, align_val, cv::BORDER_REFLECT);
    return pad_mat;
  } // un-aligned
  else
  {
    // align & padding
    const unsigned int align_h = align_val * ((h - 1) / align_val + 1);
    const unsigned int align_w = align_val * ((w - 1) / align_val + 1);
    const unsigned int pad_h = align_h - h; // >= 0
    const unsigned int pad_w = align_w - w; // >= 0
    const unsigned int target_h = h + align_val + (pad_h + align_val);
    const unsigned int target_w = w + align_val + (pad_w + align_val);

    cv::Mat pad_mat(target_h, target_w, unpad_mat.type());

    cv::copyMakeBorder(unpad_mat, pad_mat, align_val, pad_h + align_val,
                       align_val, pad_w + align_val, cv::BORDER_REFLECT);
    return pad_mat;
  }
}
