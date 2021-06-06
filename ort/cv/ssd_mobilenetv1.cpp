//
// Created by DefTruth on 2021/6/5.
//

#include "ssd_mobilenetv1.h"
#include "ort/core/ort_utils.h"

using ortcv::SSDMobileNetV1;

SSDMobileNetV1::SSDMobileNetV1(const std::string &_onnx_path, unsigned int _num_threads) :
    onnx_path(_onnx_path.data()), num_threads(_num_threads)
{
  ort_env = ort::Env(ORT_LOGGING_LEVEL_ERROR, onnx_path);
  // 0. session options
  ort::SessionOptions session_options;
  session_options.SetIntraOpNumThreads(num_threads);
  session_options.SetGraphOptimizationLevel(
      GraphOptimizationLevel::ORT_ENABLE_EXTENDED);
  session_options.SetLogSeverityLevel(4);
  // 1. session
  ort_session = new ort::Session(ort_env, onnx_path, session_options);

  ort::AllocatorWithDefaultOptions allocator;
  // 2. input name & input dims
  num_inputs = ort_session->GetInputCount();
  input_node_names.resize(num_inputs);
  // 3. initial input node dims.
  input_node_dims.push_back({batch_size, input_height, input_width, 3}); // NHWC
  input_tensor_sizes.push_back(batch_size * input_height * input_width * 3);
  input_values_handler.resize(batch_size * input_height * input_width * 3);
  for (unsigned int i = 0; i < num_inputs; ++i)
    input_node_names[i] = ort_session->GetInputName(i, allocator);
  // 4. output names & output dimms
  num_outputs = ort_session->GetOutputCount();
  output_node_names.resize(num_outputs);
  for (unsigned int i = 0; i < num_outputs; ++i)
    output_node_names[i] = ort_session->GetOutputName(i, allocator);
#if LITEORT_DEBUG
  this->print_debug_string();
#endif
}

SSDMobileNetV1::~SSDMobileNetV1()
{
  if (ort_session)
    delete ort_session;
  ort_session = nullptr;
}

void SSDMobileNetV1::print_debug_string()
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
    std::cout << "Dynamic Output " << i << ": " << output_node_names[i] << std::endl;
}

ort::Value SSDMobileNetV1::transform(const cv::Mat &mat)
{
  cv::Mat canvas = mat.clone();
  cv::resize(canvas, canvas, cv::Size(input_width, input_height));
  cv::cvtColor(canvas, canvas, cv::COLOR_BGR2RGB); // uint8 hwc

  // HWC
  std::memcpy(input_values_handler.data(), canvas.data,
              input_tensor_sizes.at(0) * sizeof(uchar));

  return ort::Value::CreateTensor<uchar>(memory_info_handler, input_values_handler.data(),
                                         input_tensor_sizes.at(0),
                                         input_node_dims.at(0).data(),
                                         input_node_dims.at(0).size());

}


void SSDMobileNetV1::detect(const cv::Mat &mat, std::vector<types::Boxf> &detected_boxes,
                            float score_threshold, float iou_threshold,
                            unsigned int topk, unsigned int nms_type)
{
  if (mat.empty()) return;
  const unsigned int img_height = mat.rows;
  const unsigned int img_width = mat.cols;
  // 1. make input tensor
  ort::Value input_tensor = this->transform(mat);
  // 2. inference nums & boxes & scores & classes.
  auto output_tensors = ort_session->Run(
      ort::RunOptions{nullptr}, input_node_names.data(),
      &input_tensor, num_inputs, output_node_names.data(),
      num_outputs
  );
  // 3. rescale & exclude.
  std::vector<types::Boxf> bbox_collection;
  this->generate_bboxes(bbox_collection, output_tensors, score_threshold, img_height, img_width);
  // 4. hard|blend nms with topk.
  this->nms(bbox_collection, detected_boxes, iou_threshold, topk, nms_type);
}

void SSDMobileNetV1::generate_bboxes(std::vector<types::Boxf> &bbox_collection,
                                     std::vector<ort::Value> &output_tensors,
                                     float score_threshold, float img_height,
                                     float img_width)
{
  ort::Value &nums = output_tensors.at(0); // (1,) float32
  ort::Value &bboxes = output_tensors.at(1); // (1,?,4)
  ort::Value &scores = output_tensors.at(2);  // (1,?)
  ort::Value &labels = output_tensors.at(3); // (1,?)

  const unsigned int num_selected = nums.At<unsigned int>({0});

  bbox_collection.clear();
  for (unsigned int i = 0; i < num_selected; ++i)
  {
    float conf = scores.At<float>({0, i});
    if (conf < score_threshold) continue;
    unsigned int label = labels.At<unsigned int>({0, i}) - 1;

    types::Boxf box;
    box.y1 = bboxes.At<float>({0, i, 0}) * (float) img_height;
    box.x1 = bboxes.At<float>({0, i, 1}) * (float) img_width;
    box.y2 = bboxes.At<float>({0, i, 2}) * (float) img_height;
    box.x2 = bboxes.At<float>({0, i, 3}) * (float) img_width;
    box.label = label;
    box.label_text = class_names[label];
    box.flag = true;
    bbox_collection.push_back(box);
  }

#if LITEORT_DEBUG
  auto boxes_dims = bboxes.GetTypeInfo().GetTensorTypeAndShapeInfo().GetShape();
  const unsigned int num_anchors = boxes_dims.at(1);
  std::cout << "detected num_anchors: " << num_anchors << "\n";
  std::cout << "generate_bboxes num: " << bbox_collection.size() << "\n";
#endif
}

void SSDMobileNetV1::nms(std::vector<types::Boxf> &input, std::vector<types::Boxf> &output,
                         float iou_threshold, unsigned int topk, unsigned int nms_type)
{
  if (nms_type == NMS::BLEND) ortcv::utils::blending_nms(input, output, iou_threshold, topk);
  else if (nms_type == NMS::OFFSET) ortcv::utils::offset_nms(input, output, iou_threshold, topk);
  else ortcv::utils::hard_nms(input, output, iou_threshold, topk);
}













































