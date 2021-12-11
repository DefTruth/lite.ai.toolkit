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
  auto padded_mat = this->padding(mat); // 0-255 int8 h+2*align_val w+2*align_val
  auto padded_mask = this->padding(mask); // 0-1.0 float32 h+2*align_val w+2*align_val

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
  if (h % align_val == 0 && w % align_val == 0)
  {
    unsigned int target_h = h + 2 * align_val;
    unsigned int target_w = w + 2 * align_val;
    cv::Mat pad_mat(target_h, target_w, unpad_mat.type());

    cv::copyMakeBorder(unpad_mat, pad_mat, align_val, align_val,
                       align_val, align_val, cv::BORDER_REFLECT);
    return pad_mat;
  } // un-aligned
  else
  {
    // align & padding
    unsigned int align_h = align_val * ((h - 1) / align_val + 1);
    unsigned int align_w = align_val * ((w - 1) / align_val + 1);
    unsigned int pad_h = align_h - h; // >= 0
    unsigned int pad_w = align_w - w; // >= 0
    unsigned int target_h = h + align_val + (pad_h + align_val);
    unsigned int target_w = w + align_val + (pad_w + align_val);

    cv::Mat pad_mat(target_h, target_w, unpad_mat.type());

    cv::copyMakeBorder(unpad_mat, pad_mat, align_val, pad_h + align_val,
                       align_val, pad_w + align_val, cv::BORDER_REFLECT);
    return pad_mat;
  }
}


void MGMatting::update_guidance_mask(cv::Mat &mask, unsigned int guidance_threshold)
{
  if (mask.type() != CV_32FC1) mask.convertTo(mask, CV_32FC1);
  const unsigned int h = mask.rows;
  const unsigned int w = mask.cols;
  if (mask.isContinuous())
  {
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
    for (unsigned int i = 0; i < h; ++i)
    {
      float *p = mask.ptr<float>(i);
      for (unsigned int j = 0; j < w; ++j)
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
                       bool remove_noise, unsigned int guidance_threshold)
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
  this->generate_matting(output_tensors, mat, content, remove_noise);
}

void MGMatting::generate_matting(std::vector<Ort::Value> &output_tensors,
                                 const cv::Mat &mat, types::MattingContent &content,
                                 bool remove_noise)
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
  float *alpha_os4_ptr = alpha_os4.GetTensorMutableData<float>();
  float *alpha_os8_ptr = alpha_os8.GetTensorMutableData<float>();

  cv::Mat alpha_os1_pred(out_h, out_w, CV_32FC1, alpha_os1_ptr);
  cv::Mat alpha_os4_pred(out_h, out_w, CV_32FC1, alpha_os4_ptr);
  cv::Mat alpha_os8_pred(out_h, out_w, CV_32FC1, alpha_os8_ptr);

  cv::Mat alpha_pred(out_h, out_w, CV_32FC1, alpha_os8_ptr);
  cv::Mat weight_os4 = this->get_unknown_tensor_from_pred(alpha_pred, 30);
  this->update_alpha_pred(alpha_pred, weight_os4, alpha_os4_pred);
  cv::Mat weight_os1 = this->get_unknown_tensor_from_pred(alpha_pred, 15);
  this->update_alpha_pred(alpha_pred, weight_os1, alpha_os1_pred);
  // post process
  if (remove_noise) this->remove_small_connected_area(alpha_pred);

  cv::Mat mat_copy;
  mat.convertTo(mat_copy, CV_32FC3);
  // cv::Mat pred_alpha_mat(out_h, out_w, CV_32FC1, alpha_os1_ptr);
  cv::Mat pmat = alpha_pred(cv::Rect(align_val, align_val, w, h)).clone();

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

  content.pha_mat = pmat;
  cv::merge(fgr_channel_mats, content.fgr_mat);
  cv::merge(merge_channel_mats, content.merge_mat);

  content.fgr_mat.convertTo(content.fgr_mat, CV_8UC3);
  content.merge_mat.convertTo(content.merge_mat, CV_8UC3);

  content.flag = true;
}

// https://github.com/yucornetto/MGMatting/issues/11
// https://github.com/yucornetto/MGMatting/blob/main/code-base/utils/util.py#L225
cv::Mat MGMatting::get_unknown_tensor_from_pred(const cv::Mat &alpha_pred, unsigned int rand_width)
{
  const unsigned int h = alpha_pred.rows;
  const unsigned int w = alpha_pred.cols;
  const unsigned int data_size = h * w;
  cv::Mat uncertain_area(h, w, CV_32FC1, cv::Scalar(1.0f)); // continuous
  const float *pred_ptr = (float *) alpha_pred.data;
  float *uncertain_ptr = (float *) uncertain_area.data;
  // threshold
  if (alpha_pred.isContinuous() && uncertain_area.isContinuous())
  {
    for (unsigned int i = 0; i < data_size; ++i)
      if ((pred_ptr[i] < 1.0f / 255.0f) || (pred_ptr[i] > 1.0f - 1.0f / 255.0f))
        uncertain_ptr[i] = 0.f;
  } //
  else
  {
    for (unsigned int i = 0; i < h; ++i)
    {
      const float *pred_row_ptr = alpha_pred.ptr<float>(i);
      float *uncertain_row_ptr = uncertain_area.ptr<float>(i);
      for (unsigned int j = 0; j < w; ++j)
      {
        if ((pred_row_ptr[j] < 1.0f / 255.0f) || (pred_row_ptr[j] > 1.0f - 1.0f / 255.0f))
          uncertain_row_ptr[j] = 0.f;
      }
    }
  }
  // dilate
  unsigned int size = rand_width / 2;
  auto kernel = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(size, size));
  cv::dilate(uncertain_area, uncertain_area, kernel);

  // weight
  cv::Mat weight(h, w, CV_32FC1, uncertain_area.data); // ref only, zero copy.
  float *weight_ptr = (float *) weight.data;
  if (weight.isContinuous())
  {
    for (unsigned int i = 0; i < data_size; ++i)
      if (weight_ptr[i] != 1.0f) weight_ptr[i] = 0;
  } //
  else
  {
    for (unsigned int i = 0; i < h; ++i)
    {
      float *weight_row_ptr = weight.ptr<float>(i);
      for (unsigned int j = 0; j < w; ++j)
        if (weight_row_ptr[j] != 1.0f) weight_row_ptr[j] = 0.f;

    }
  }

  return weight;
}

void MGMatting::update_alpha_pred(cv::Mat &alpha_pred, const cv::Mat &weight, const cv::Mat &other_alpha_pred)
{
  const unsigned int h = alpha_pred.rows;
  const unsigned int w = alpha_pred.cols;
  const unsigned int data_size = h * w;
  const float *weight_ptr = (float *) weight.data;
  float *mutable_alpha_ptr = (float *) alpha_pred.data;
  const float *other_alpha_ptr = (float *) other_alpha_pred.data;

  if (alpha_pred.isContinuous() && weight.isContinuous() && other_alpha_pred.isContinuous())
  {
    for (unsigned int i = 0; i < data_size; ++i)
      if (weight_ptr[i] > 0.f) mutable_alpha_ptr[i] = other_alpha_ptr[i];
  } //
  else
  {
    for (unsigned int i = 0; i < h; ++i)
    {
      const float *weight_row_ptr = weight.ptr<float>(i);
      float *mutable_alpha_row_ptr = alpha_pred.ptr<float>(i);
      const float *other_alpha_row_ptr = other_alpha_pred.ptr<float>(i);
      for (unsigned int j = 0; j < w; ++j)
        if (weight_row_ptr[j] > 0.f) mutable_alpha_row_ptr[j] = other_alpha_row_ptr[j];
    }
  }
}

// https://github.com/yucornetto/MGMatting/blob/main/code-base/utils/util.py#L208
void MGMatting::remove_small_connected_area(cv::Mat &alpha_pred)
{
  cv::Mat gray, binary;
  alpha_pred.convertTo(gray, CV_8UC1, 255.f);
  // 255 * 0.05 ~ 13
  cv::threshold(gray, binary, 13, 255, cv::THRESH_BINARY);
  // morphologyEx with OPEN operation to remove noise first.
  auto kernel = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(3, 3), cv::Point(-1, -1));
  cv::morphologyEx(binary, binary, cv::MORPH_OPEN, kernel);
  // Computationally connected domain
  cv::Mat labels = cv::Mat::zeros(alpha_pred.size(), CV_32S);
  cv::Mat stats, centroids;
  int num_labels = cv::connectedComponentsWithStats(binary, labels, stats, centroids, 8, 4);
  if (num_labels <= 1) return; // no noise, skip.
  // find max connected area, 0 is background
  int max_connected_id = 1; // 1,2,...
  int max_connected_area = stats.at<int>(max_connected_id, cv::CC_STAT_AREA);
  for (int i = 1; i < num_labels; ++i)
  {
    int tmp_connected_area = stats.at<int>(i, cv::CC_STAT_AREA);
    if (tmp_connected_area > max_connected_area)
    {
      max_connected_area = tmp_connected_area;
      max_connected_id = i;
    }
  }
  std::cout << max_connected_id << std::endl;
  std::cout << num_labels << std::endl;
  const int h = alpha_pred.rows;
  const int w = alpha_pred.cols;
  // remove small connected area.
  for (int i = 0; i < h; ++i)
  {
    int *label_row_ptr = labels.ptr<int>(i);
    float *alpha_row_ptr = alpha_pred.ptr<float>(i);
    for (int j = 0; j < w; ++j)
    {
      if (label_row_ptr[j] != max_connected_id)
        alpha_row_ptr[j] = 0.f;
    }
  }
}

void MGMatting::update_dynamic_shape(unsigned int img_height, unsigned int img_width)
{
  unsigned int h = img_height;
  unsigned int w = img_width;
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
    unsigned int align_h = align_val * ((h - 1) / align_val + 1);
    unsigned int align_w = align_val * ((w - 1) / align_val + 1);
    unsigned int pad_h = align_h - h; // >= 0
    unsigned int pad_w = align_w - w; // >= 0
    dynamic_input_height = h + align_val + (pad_h + align_val);
    dynamic_input_width = w + align_val + (pad_w + align_val);
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