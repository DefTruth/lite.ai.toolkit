//
// Created by DefTruth on 2021/12/5.
//

#include "mg_matting.h"
#include "lite/ort/core/ort_utils.h"
#include "lite/utils.h"

using ortcv::MGMatting;

MGMatting::MGMatting(const std::string &_onnx_path, unsigned int _num_threads) :
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
  dynamic_image_values_handler.resize(dynamic_input_image_size);
  dynamic_mask_values_handler.resize(dynamic_input_mask_size);
#if LITEORT_DEBUG
  this->print_debug_string();
#endif
}

MGMatting::~MGMatting()
{
  if (ort_session)
    delete ort_session;
  ort_session = nullptr;
}

void MGMatting::print_debug_string()
{
  std::cout << "LITEORT_DEBUG LogId: " << onnx_path << "\n";
  std::cout << "=============== Inputs ==============\n";
  std::cout << "Dynamic Input: " << "image" << "Init ["
            << 1 << "," << 3 << "," << dynamic_input_height
            << "," << dynamic_input_width << "]\n";
  std::cout << "Dynamic Input: " << "mask" << "Init ["
            << 1 << "," << 1 << "," << dynamic_input_height
            << "," << dynamic_input_width << "]\n";
  std::cout << "=============== Outputs ==============\n";
  for (unsigned int i = 0; i < num_outputs; ++i)
    std::cout << "Dynamic Output " << i << ": " << output_node_names[i] << std::endl;
}

std::vector<Ort::Value> MGMatting::transform(const cv::Mat &mat, const cv::Mat &mask)
{
  auto padded_mat = this->padding(mat); // 0-255 int8 h+2*pad_val w+2*pad_val
  auto padded_mask = this->padding(mask); // 0-1.0 float32 h+2*pad_val w+2*pad_val

  cv::Mat canvas;
  cv::cvtColor(padded_mat, canvas, cv::COLOR_BGR2RGB);
  canvas.convertTo(canvas, CV_32FC3, 1.f / 255.f, 0.f); // (0.,1.)
  ortcv::utils::transform::normalize_inplace(canvas, mean_vals, scale_vals);

  // convert to tensor.
  std::vector<Ort::Value> input_tensors;

  input_tensors.emplace_back(ortcv::utils::transform::create_tensor(
      canvas, dynamic_input_image_dims, memory_info_handler,
      dynamic_image_values_handler, ortcv::utils::transform::CHW
  )); // image 1x3xhxw

  input_tensors.emplace_back(ortcv::utils::transform::create_tensor(
      padded_mask, dynamic_input_mask_dims, memory_info_handler,
      dynamic_mask_values_handler, ortcv::utils::transform::CHW
  )); // mask 1x1xhxw

  return input_tensors;
}

cv::Mat MGMatting::padding(const cv::Mat &unpad_mat)
{
  const unsigned int h = unpad_mat.rows;
  const unsigned int w = unpad_mat.cols;

  // aligned
  if (h % pad_val == 0 && w % pad_val == 0)
  {
    unsigned int target_h = h + 2 * pad_val;
    unsigned int target_w = w + 2 * pad_val;
    cv::Mat pad_mat(target_h, target_w, unpad_mat.type());

    cv::copyMakeBorder(unpad_mat, pad_mat, pad_val, pad_val,
                       pad_val, pad_val, cv::BORDER_REFLECT);
    return pad_mat;
  } // un-aligned
  else
  {
    // align & padding
    unsigned int align_h = pad_val * ((h - 1) / pad_val + 1);
    unsigned int align_w = pad_val * ((w - 1) / pad_val + 1);
    unsigned int pad_h = align_h - h; // >= 0
    unsigned int pad_w = align_w - w; // >= 0
    unsigned int target_h = h + pad_val + (pad_h + pad_val);
    unsigned int target_w = w + pad_val + (pad_w + pad_val);

    cv::Mat pad_mat(target_h, target_w, unpad_mat.type());

    cv::copyMakeBorder(unpad_mat, pad_mat, pad_val, pad_h + pad_val,
                       pad_val, pad_w + pad_val, cv::BORDER_REFLECT);
    return pad_mat;
  }
}


void MGMatting::update_guidance_mask(cv::Mat &mask, unsigned int guidance_threshold)
{
  if (mask.type() != CV_32FC1) mask.convertTo(mask, CV_32FC1);
  if (mask.isContinuous())
  {
    const unsigned int h = mask.rows;
    const unsigned int w = mask.cols;
    const unsigned int data_size = h * w * 1;
    float *mutable_data_ptr = (float *) mask.data;
    float guidance_threshold_ = (float) guidance_threshold;
    for (unsigned int i = 0; i < data_size; ++i)
    {
      if (mutable_data_ptr[i] >= guidance_threshold_)
        mutable_data_ptr[i] = 1.0f;
      else
        mutable_data_ptr[i] = 0.0f;
    }
  } //
  else
  {
    float guidance_threshold_ = (float) guidance_threshold;
    for (unsigned int i = 0; i < mask.rows; ++i)
    {
      float *p = mask.ptr<float>(i);
      for (unsigned int j = 0; j < mask.cols; ++j)
      {
        if (p[j] >= guidance_threshold_)
          p[j] = 1.0;
        else
          p[j] = 0.;
      }
    }
  }
}

void MGMatting::detect(const cv::Mat &mat, cv::Mat &mask, types::MattingContent &content,
                       unsigned int guidance_threshold)
{
  if (mat.empty() || mask.empty()) return;
  const unsigned int img_height = mat.rows;
  const unsigned int img_width = mat.cols;
  this->update_dynamic_shape(img_height, img_width);
  this->update_guidance_mask(mask, guidance_threshold); // -> float32 hw1 0~1.0

  // 1. make input tensors, image, mask
  std::vector<Ort::Value> input_tensors = this->transform(mat, mask);
  // 2. inference, fgr, pha, rxo.
  auto output_tensors = ort_session->Run(
      Ort::RunOptions{nullptr}, input_node_names.data(),
      input_tensors.data(), num_inputs, output_node_names.data(),
      num_outputs
  );
  // 3. generate matting
  this->generate_matting(output_tensors, mat, content);
}

void MGMatting::generate_matting(std::vector<Ort::Value> &output_tensors,
                                 const cv::Mat &mat, types::MattingContent &content)
{
  Ort::Value &alpha_os1 = output_tensors.at(0); // (1,1,h+?,w+?)
  Ort::Value &alpha_os4 = output_tensors.at(1); // (1,1,h+?,w+?)
  Ort::Value &alpha_os8 = output_tensors.at(2); // (1,1,h+?,w+?)
  const unsigned int h = mat.rows;
  const unsigned int w = mat.cols;

  // TODO: add post-process as official python implementation.
  // https://github.com/yucornetto/MGMatting/blob/main/code-base/infer.py
  auto output_dims = alpha_os1.GetTypeInfo().GetTensorTypeAndShapeInfo().GetShape();
  const unsigned int out_h = output_dims.at(2);
  const unsigned int out_w = output_dims.at(3);
  float *alpha_os1_ptr = alpha_os1.GetTensorMutableData<float>();

  cv::Mat pred_alpha_mat(out_h, out_w, CV_32FC1, alpha_os1_ptr);
  content.pha_mat = pred_alpha_mat(cv::Rect(pad_val, pad_val, w, h)).clone();
  content.fgr_mat = mat.mul(content.pha_mat);
  cv::Mat bgmat(h, w, CV_32FC3, cv::Scalar(153.f, 255.f, 120.f)); // background mat
  cv::Mat rest = 1. - content.pha_mat;
  content.merge_mat = content.fgr_mat + bgmat.mul(rest);

  content.fgr_mat.convertTo(content.fgr_mat, CV_8UC3);
  content.merge_mat.convertTo(content.merge_mat, CV_8UC3);

  content.flag = true;
}

void MGMatting::update_dynamic_shape(unsigned int img_height, unsigned int img_width)
{
  unsigned int h = img_height;
  unsigned int w = img_width;
  // update dynamic input dims
  if (h % pad_val == 0 && w % pad_val == 0)
  {
    // aligned
    dynamic_input_height = h + 2 * pad_val;
    dynamic_input_width = w + 2 * pad_val;

  } // un-aligned
  else
  {
    // align first
    unsigned int align_h = pad_val * ((h - 1) / pad_val + 1);
    unsigned int align_w = pad_val * ((w - 1) / pad_val + 1);
    unsigned int pad_h = align_h - h; // >= 0
    unsigned int pad_w = align_w - w; // >= 0
    dynamic_input_height = h + pad_val + (pad_h + pad_val);
    dynamic_input_width = w + pad_val + (pad_w + pad_val);
  }

  // update dims info
  dynamic_input_image_dims.at(2) = dynamic_input_height;
  dynamic_input_image_dims.at(3) = dynamic_input_width;
  dynamic_input_mask_dims.at(2) = dynamic_input_height;
  dynamic_input_mask_dims.at(3) = dynamic_input_width;

  dynamic_input_image_size = 1 * 3 * dynamic_input_height * dynamic_input_width;
  dynamic_input_mask_size = 1 * 1 * dynamic_input_height * dynamic_input_width;
  dynamic_image_values_handler.resize(dynamic_input_image_size);
  dynamic_mask_values_handler.resize(dynamic_input_mask_size);
}