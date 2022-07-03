//
// Created by DefTruth on 2022/3/28.
//

#include "modnet_dyn.h"
#include "lite/ort/core/ort_utils.h"
#include "lite/utils.h"

using ortcv::MODNetDyn;

MODNetDyn::MODNetDyn(const std::string &_onnx_path, unsigned int _num_threads) :
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
  input_node_names[0] = ort_session->GetInputName(0, allocator);
  // 3. initial input node dims.
  dynamic_input_node_dims.push_back({1, 3, dynamic_input_height, dynamic_input_width});
  dynamic_input_tensor_size = 1 * 3 * dynamic_input_height * dynamic_input_width;
  dynamic_input_values_handler.resize(dynamic_input_tensor_size);
  // 4. output names & output dimms
  num_outputs = ort_session->GetOutputCount();
  output_node_names.resize(num_outputs);
  for (unsigned int i = 0; i < num_outputs; ++i)
    output_node_names[i] = ort_session->GetOutputName(i, allocator);
#if LITEORT_DEBUG
  this->print_debug_string();
#endif
}

MODNetDyn::~MODNetDyn()
{
  if (ort_session)
    delete ort_session;
  ort_session = nullptr;
}

void MODNetDyn::print_debug_string()
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

Ort::Value MODNetDyn::transform(const cv::Mat &mat)
{
  auto canvas = this->padding(mat);

  cv::cvtColor(canvas, canvas, cv::COLOR_BGR2RGB);
  ortcv::utils::transform::normalize_inplace(canvas, mean_val, scale_val);

  return ortcv::utils::transform::create_tensor(
      canvas, dynamic_input_node_dims.at(0), memory_info_handler,
      dynamic_input_values_handler, ortcv::utils::transform::CHW
  );
}

void MODNetDyn::detect(const cv::Mat &mat, types::MattingContent &content, bool remove_noise,
                       bool minimum_post_process)
{
  if (mat.empty()) return;
  const unsigned int img_height = mat.rows;
  const unsigned int img_width = mat.cols;
  // align input size with 32 first
  this->update_dynamic_shape(img_height, img_width);

  // 1. make input tensor
  Ort::Value input_tensor = this->transform(mat);
  // 2. inference
  auto output_tensors = ort_session->Run(
      Ort::RunOptions{nullptr}, input_node_names.data(),
      &input_tensor, 1, output_node_names.data(), num_outputs
  );
  // 3. generate matting
  this->generate_matting(output_tensors, mat, content, remove_noise, minimum_post_process);
}

void MODNetDyn::generate_matting(std::vector<Ort::Value> &output_tensors,
                                 const cv::Mat &mat, types::MattingContent &content,
                                 bool remove_noise, bool minimum_post_process)
{
  Ort::Value &output = output_tensors.at(0); // (1,1,out_h,out_w) 0~1
  const unsigned int h = mat.rows;
  const unsigned int w = mat.cols;

  auto output_dims = output.GetTypeInfo().GetTensorTypeAndShapeInfo().GetShape();
  const unsigned int out_h = output_dims.at(2);
  const unsigned int out_w = output_dims.at(3);

  float *output_ptr = output.GetTensorMutableData<float>();

  cv::Mat alpha_pred(out_h, out_w, CV_32FC1, output_ptr);
  // need clone to allocate a new continuous memory.
  cv::Mat pmat = alpha_pred(cv::Rect(align_val, align_val, w, h)).clone(); // allocated
  if (remove_noise) lite::utils::remove_small_connected_area(pmat, 0.05f);

  content.pha_mat = pmat;

  if (!minimum_post_process)
  {
    // MODNet only predict Alpha, no fgr. So,
    // the fake fgr and merge mat may not need,
    // let the fgr mat and merge mat empty to
    // Speed up the post processes.
    cv::Mat mat_copy;
    mat.convertTo(mat_copy, CV_32FC3);
    // merge mat and fgr mat may not need
    std::vector<cv::Mat> mat_channels;
    cv::split(mat_copy, mat_channels);
    cv::Mat bmat = mat_channels.at(0);
    cv::Mat gmat = mat_channels.at(1);
    cv::Mat rmat = mat_channels.at(2); // ref only, zero-copy.
    bmat = bmat.mul(pmat);
    gmat = gmat.mul(pmat);
    rmat = rmat.mul(pmat);
    cv::Mat rest = 1.f - pmat;
    cv::Mat mbmat = bmat.mul(pmat) + rest * 153.f;
    cv::Mat mgmat = gmat.mul(pmat) + rest * 255.f;
    cv::Mat mrmat = rmat.mul(pmat) + rest * 120.f;
    std::vector<cv::Mat> fgr_channel_mats, merge_channel_mats;
    fgr_channel_mats.push_back(bmat);
    fgr_channel_mats.push_back(gmat);
    fgr_channel_mats.push_back(rmat);
    merge_channel_mats.push_back(mbmat);
    merge_channel_mats.push_back(mgmat);
    merge_channel_mats.push_back(mrmat);

    cv::merge(fgr_channel_mats, content.fgr_mat); // allocated
    cv::merge(merge_channel_mats, content.merge_mat); // allocated

    content.fgr_mat.convertTo(content.fgr_mat, CV_8UC3);
    content.merge_mat.convertTo(content.merge_mat, CV_8UC3);
  }

  content.flag = true;
}

void MODNetDyn::update_dynamic_shape(unsigned int img_height, unsigned int img_width)
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
  dynamic_input_tensor_size = 1 * 3 * dynamic_input_height * dynamic_input_width;
  dynamic_input_values_handler.resize(dynamic_input_tensor_size);
}

cv::Mat MODNetDyn::padding(const cv::Mat &unpad_mat)
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




