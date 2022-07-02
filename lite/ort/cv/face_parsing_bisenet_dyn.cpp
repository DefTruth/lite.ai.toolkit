//
// Created by DefTruth on 2022/6/29.
//

#include "face_parsing_bisenet_dyn.h"
#include "lite/ort/core/ort_utils.h"
#include "lite/utils.h"

using ortcv::FaceParsingBiSeNetDyn;

FaceParsingBiSeNetDyn::FaceParsingBiSeNetDyn(const std::string &_onnx_path, unsigned int _num_threads) :
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

FaceParsingBiSeNetDyn::~FaceParsingBiSeNetDyn()
{
  if (ort_session)
    delete ort_session;
  ort_session = nullptr;
}

void FaceParsingBiSeNetDyn::print_debug_string()
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

Ort::Value FaceParsingBiSeNetDyn::transform(const cv::Mat &mat)
{
  auto canvas = this->padding(mat);

  cv::cvtColor(canvas, canvas, cv::COLOR_BGR2RGB);
  ortcv::utils::transform::normalize_inplace(canvas, mean_vals, scale_vals);

  return ortcv::utils::transform::create_tensor(
      canvas, dynamic_input_node_dims.at(0), memory_info_handler,
      dynamic_input_values_handler, ortcv::utils::transform::CHW
  );
}

void FaceParsingBiSeNetDyn::detect(const cv::Mat &mat, types::FaceParsingContent &content,
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
  // 3. generate mask
  this->generate_mask(output_tensors, mat, content, minimum_post_process);
}

static inline uchar __argmax_find(float *mutable_ptr, const unsigned int &step)
{
  std::vector<float> logits(19, 0.f);
  for (unsigned int i = 0; i < 19; ++i)
    logits[i] = *(mutable_ptr + i * step);
  uchar label = 0;
  float max_logit = logits[0];
  for (unsigned int i = 1; i < 19; ++i)
  {
    if (logits[i] > max_logit)
    {
      max_logit = logits[i];
      label = (uchar) i;
    }
  }
  return label;
}

static const uchar part_colors[20][3] = {
    {255, 0,   0},
    {255, 85,  0},
    {255, 170, 0},
    {255, 0,   85},
    {255, 0,   170},
    {0,   255, 0},
    {85,  255, 0},
    {170, 255, 0},
    {0,   255, 85},
    {0,   255, 170},
    {0,   0,   255},
    {85,  0,   255},
    {170, 0,   255},
    {0,   85,  255},
    {0,   170, 255},
    {255, 255, 0},
    {255, 255, 85},
    {255, 255, 170},
    {255, 0,   255},
    {255, 85,  255}
};

void FaceParsingBiSeNetDyn::generate_mask(std::vector<Ort::Value> &output_tensors, const cv::Mat &mat,
                                          types::FaceParsingContent &content,
                                          bool minimum_post_process)
{
  Ort::Value &output = output_tensors.at(0); // (1,19,h,w)
  const unsigned int h = mat.rows;
  const unsigned int w = mat.cols;

  auto output_dims = output.GetTypeInfo().GetTensorTypeAndShapeInfo().GetShape();
  const unsigned int out_h = output_dims.at(2);
  const unsigned int out_w = output_dims.at(3);
  const unsigned int channel_step = out_h * out_w;

  float *output_ptr = output.GetTensorMutableData<float>();
  std::vector<uchar> elements(channel_step, 0); // allocate
  for (unsigned int i = 0; i < channel_step; ++i)
    elements[i] = __argmax_find(output_ptr + i, channel_step);

  cv::Mat label_pred(out_h, out_w, CV_8UC1, elements.data()); // ref only
  // need clone to allocate a new continuous memory.
  cv::Mat label = label_pred(cv::Rect(align_val, align_val, w, h)).clone();
  if (label.type() != CV_8UC1) label.convertTo(label, CV_8UC1, 1., 0.);

  if (!minimum_post_process)
  {
    // FaceParsingBiSeNet only predict integer label mask,
    // no fgr. So, the fake fgr and merge mat may not need,
    // let the fgr mat and merge mat empty to
    // Speed up the post processes.
    const uchar *label_ptr = label.data;
    cv::Mat color_mat(h, w, CV_8UC3, cv::Scalar(255, 255, 255));
    for (unsigned int i = 0; i < color_mat.rows; ++i)
    {
      cv::Vec3b *p = color_mat.ptr<cv::Vec3b>(i);
      for (unsigned int j = 0; j < color_mat.cols; ++j)
      {
        if (label_ptr[i * w + j] == 0) continue;
        p[j][0] = part_colors[label_ptr[i * w + j]][0];
        p[j][1] = part_colors[label_ptr[i * w + j]][1];
        p[j][2] = part_colors[label_ptr[i * w + j]][2];
      }
    }
    cv::addWeighted(mat, 0.4, color_mat, 0.6, 0., content.merge);
  }
  content.label = label; // no need clone, already cloned.
  content.flag = true;
}

void FaceParsingBiSeNetDyn::update_dynamic_shape(unsigned int img_height, unsigned int img_width)
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

cv::Mat FaceParsingBiSeNetDyn::padding(const cv::Mat &unpad_mat)
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





























