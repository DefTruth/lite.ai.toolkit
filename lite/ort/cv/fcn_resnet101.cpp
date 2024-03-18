//
// Created by DefTruth on 2021/6/14.
//

#include "fcn_resnet101.h"
#include "lite/ort/core/ort_utils.h"
#include "lite/utils.h"

using ortcv::FCNResNet101;

FCNResNet101::FCNResNet101(const std::string &_onnx_path, unsigned int _num_threads) :
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
  Ort::AllocatorWithDefaultOptions allocator;
  // 2. input name & input dims
  input_node_names.resize(num_inputs); // num_inputs=1
  input_node_names_.resize(num_inputs); // num_inputs=1
  input_node_names_[0] = OrtCompatiableGetInputName(0, allocator, ort_session);
  input_node_names[0] = input_node_names_[0].data();
  // 3. initial input node dims.
  dynamic_input_node_dims.push_back({1, 3, dynamic_input_height, dynamic_input_width});
  dynamic_input_tensor_size = 1 * 3 * dynamic_input_height * dynamic_input_width;
  dynamic_input_values_handler.resize(dynamic_input_tensor_size);
  // 4. output names & output dimms
  num_outputs = ort_session->GetOutputCount();
  output_node_names.resize(num_outputs);
  output_node_names_.resize(num_outputs);
  for (unsigned int i = 0; i < num_outputs; ++i) {
    output_node_names_[i] = OrtCompatiableGetOutputName(i, allocator, ort_session);
    output_node_names[i] = output_node_names_[i].data();
  }
#if LITEORT_DEBUG
  this->print_debug_string();
#endif
}

FCNResNet101::~FCNResNet101()
{
  if (ort_session)
    delete ort_session;
  ort_session = nullptr;
}

void FCNResNet101::print_debug_string()
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

Ort::Value FCNResNet101::transform(const cv::Mat &mat)
{
  cv::Mat canvas;
  const unsigned int img_height = mat.rows;
  const unsigned int img_width = mat.cols;

  // update dynamic input dims
  dynamic_input_height = img_height;
  dynamic_input_width = img_width;
  dynamic_input_node_dims.at(0).at(2) = dynamic_input_height;
  dynamic_input_node_dims.at(0).at(3) = dynamic_input_width;
  dynamic_input_tensor_size = 1 * 3 * dynamic_input_height * dynamic_input_width;
  dynamic_input_values_handler.resize(dynamic_input_tensor_size);

  cv::cvtColor(mat, canvas, cv::COLOR_BGR2RGB);
  canvas.convertTo(canvas, CV_32FC3, 1.f / 255.f, 0.f); // (0.,1.)

  ortcv::utils::transform::normalize_inplace(canvas, mean_vals, scale_vals);

  return ortcv::utils::transform::create_tensor(
      canvas, dynamic_input_node_dims.at(0), memory_info_handler,
      dynamic_input_values_handler, ortcv::utils::transform::CHW
  );

}

void FCNResNet101::detect(const cv::Mat &mat, types::SegmentContent &content)
{
  if (mat.empty()) return;
  // 1. make input tensor
  Ort::Value input_tensor = this->transform(mat);
  // 2. inference out(scores) (1,21=1+20,h,w) pixel to pixel
  auto output_tensors = ort_session->Run(
      Ort::RunOptions{nullptr}, input_node_names.data(),
      &input_tensor, num_inputs, output_node_names.data(),
      num_outputs
  );
  // 3. post process.
  Ort::Value &scores = output_tensors.at(0); // (1,21,h,w)
  auto scores_dims = scores.GetTypeInfo().GetTensorTypeAndShapeInfo().GetShape();
  const unsigned int output_classes = scores_dims.at(1);
  const unsigned int output_height = scores_dims.at(2);
  const unsigned int output_width = scores_dims.at(3);

  // time cost!
  content.names_map.clear();
  content.class_mat = cv::Mat(output_height, output_width, CV_8UC1, cv::Scalar(0));
  content.color_mat = mat.clone();

  for (unsigned int i = 0; i < output_height; ++i)
  {

    uchar *p_class = content.class_mat.ptr<uchar>(i);
    cv::Vec3b *p_color = content.color_mat.ptr<cv::Vec3b>(i);

    for (unsigned int j = 0; j < output_width; ++j)
    {
      // argmax
      unsigned int max_label = 0;
      float max_conf = scores.At<float>({0, 0, i, j});

      for (unsigned int l = 0; l < output_classes; ++l)
      {
        float conf = scores.At<float>({0, l, i, j});
        if (conf > max_conf)
        {
          max_conf = conf;
          max_label = l;
        }
      }

      if (max_label == 0) continue;

      // assign label for pixel(i,j)
      p_class[j] = cv::saturate_cast<uchar>(max_label);
      // assign color for detected class at pixel(i,j).
      p_color[j][0] = cv::saturate_cast<uchar>((max_label % 10) * 20);
      p_color[j][1] = cv::saturate_cast<uchar>((max_label % 5) * 40);
      p_color[j][2] = cv::saturate_cast<uchar>((max_label % 10) * 20 );
      // assign names map
      content.names_map[max_label] = class_names[max_label - 1]; // max_label >= 1
    }

  }

  content.flag = true;
}