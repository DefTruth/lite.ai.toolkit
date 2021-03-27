//
// Created by YanJun Qiu on 2021/3/14.
//

#include "ultraface.h"

using ortcv::UltraBox;
using ortcv::UltraFace;

UltraFace::UltraFace(const std::string &_onnx_path, int _input_height, int _input_width,
                     unsigned int _num_threads) : onnx_path(_onnx_path.data()),
                                                  input_height(_input_height),
                                                  input_width(_input_width),
                                                  num_threads(_num_threads) {
  ort_env = ort::Env(ORT_LOGGING_LEVEL_ERROR, "ultraface-onnx");
  // 0. session options
  ort::SessionOptions session_options;
  session_options.SetIntraOpNumThreads(num_threads);
  session_options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_EXTENDED);
  session_options.SetLogSeverityLevel(4);
  // 1. session
  ort_session = new ort::Session(ort_env, onnx_path, session_options);

  ort::AllocatorWithDefaultOptions allocator;
  // 2. input name & input dims
  input_name = ort_session->GetInputName(0, allocator);
  input_node_names.resize(1);
  input_node_names[0] = input_name;
  // 3. type info.
  ort::TypeInfo type_info = ort_session->GetInputTypeInfo(0);
  auto tensor_info = type_info.GetTensorTypeAndShapeInfo();
  input_tensor_size = 1;
  input_node_dims = tensor_info.GetShape();
  for (unsigned int i = 0; i < input_node_dims.size(); ++i)
    input_tensor_size *= input_node_dims.at(i);
  input_tensor_values.resize(input_tensor_size);
  // 4. output names & output dimms
  num_outputs = ort_session->GetOutputCount();
  output_node_names.resize(num_outputs);
  for (unsigned int i = 0; i < num_outputs; ++i) {
    output_node_names[i] = ort_session->GetOutputName(i, allocator);
    ort::TypeInfo output_type_info = ort_session->GetOutputTypeInfo(i);
    auto output_tensor_info = output_type_info.GetTensorTypeAndShapeInfo();
    auto output_dims = output_tensor_info.GetShape();
    output_node_dims.push_back(output_dims);
  }
#if LITEORT_DEBUG
  std::cout << "=============== Input-Dims ==============\n";
  for (unsigned int i = 0; i < input_node_dims.size(); ++i)
    std::cout << "input_node_dims: " << input_node_dims.at(i) << "\n";
  std::cout << "=============== Output-Dims ==============\n";
  for (unsigned int i = 0; i < num_outputs; ++i)
    for (unsigned int j = 0; j < output_node_dims.at(i).size(); ++j)
      std::cout << "Output: " << i << " Name: "
                << output_node_names.at(i) << " Dim: " << j << " :"
                << output_node_dims.at(i).at(j) << std::endl;
#endif
}

UltraFace::~UltraFace() {
  if (ort_session)
    delete ort_session;
  ort_session = nullptr;
}

void UltraFace::preprocess(const cv::Mat &mat) {
  cv::Mat canva = mat.clone();
  cv::cvtColor(canva, canva, cv::COLOR_BGR2RGB);

  cv::Mat resize_norm;
  cv::resize(canva, canva, cv::Size(input_width, input_height)); // (640,480) | (320,240)
  canva.convertTo(resize_norm, CV_32FC3); // Note !!! should convert to float32 firstly.
  resize_norm = (resize_norm - mean_val) * scale_val; // then, normalize. MatExpr WARN:0

  std::vector<cv::Mat> channels;
  cv::split(resize_norm, channels);
  std::vector<float> channel_values;
  channel_values.resize(input_height * input_width);
  for (int i = 0; i < channels.size(); ++i) {
    channel_values.clear();
    channel_values = channels.at(i).reshape(1, 1); // flatten
    std::memcpy(input_tensor_values.data() + i * (input_height * input_width),
                channel_values.data(),
                input_height * input_width * sizeof(float)); // CXHXW
  }
}

void UltraFace::detect(const cv::Mat &mat, std::vector<UltraBox> &detected_boxes,
                       float score_threshold, float iou_threshold, int topk) {
  if (mat.empty()) return;
  this->preprocess(mat);
  float img_height = static_cast<float>(mat.rows);
  float img_width = static_cast<float>(mat.cols);

  // 1. make input tensor
  ort::Value input_tensor = ort::Value::CreateTensor<float>(
      memory_info, input_tensor_values.data(),
      input_tensor_size, input_node_dims.data(),
      4
  );
  // 2. inference scores & boxes.
  auto output_tensors = ort_session->Run(
      ort::RunOptions{nullptr}, input_node_names.data(),
      &input_tensor, 1, output_node_names.data(), num_outputs
  );
  // 3. rescale & exclude.
  std::vector<UltraBox> bbox_collection;
  this->generate_bboxes(bbox_collection, output_tensors, score_threshold, img_height, img_width);
  // 4. hard nms with topk.
  this->hard_nms(bbox_collection, detected_boxes, iou_threshold, topk);
}

void UltraFace::generate_bboxes(std::vector<UltraBox> &bbox_collection,
                                std::vector<ort::Value> &output_tensors,
                                float score_threshold, float img_height,
                                float img_width) {
  ort::Value &scores = output_tensors.at(0);
  ort::Value &boxes = output_tensors.at(1);
  auto scores_dims = output_node_dims.at(0); // (1,n,2)
  auto boxes_dims = output_node_names.at(1); // (1,n,4) x1,y1,x2,y2
  const unsigned int num_anchors = scores_dims.at(1); // n = 17640 (640x480)

  bbox_collection.clear();
  for (unsigned int i = 0; i < num_anchors; ++i) {
    float confidence = scores.At<float>({0, i, 1});
    if (confidence < score_threshold) continue;
    UltraBox box;
    box.x1 = boxes.At<float>({0, i, 0}) * img_width;
    box.y1 = boxes.At<float>({0, i, 1}) * img_height;
    box.x2 = boxes.At<float>({0, i, 2}) * img_width;
    box.y2 = boxes.At<float>({0, i, 3}) * img_height;
    box.score = confidence;
    bbox_collection.push_back(box);
  }
#if LITEORT_DEBUG
  std::cout << "detected num_anchors: " << num_anchors << "\n";
  std::cout << "generate_bboxes num: " << bbox_collection.size() << "\n";
#endif
}

// reference: https://github.com/Linzaer/Ultra-Light-Fast-Generic-Face-Detector-1MB/
//            blob/master/ncnn/src/UltraFace.cpp
void UltraFace::hard_nms(std::vector<UltraBox> &input,
                         std::vector<UltraBox> &output,
                         float iou_threshold, int topk) {
  if (input.empty()) return;
  std::sort(input.begin(), input.end(),
            [](const UltraBox &a, const UltraBox &b) { return a.score > b.score; });
  const unsigned int box_num = input.size();
  std::vector<int> merged(box_num, 0);

  int count = 0;

  for (unsigned int i = 0; i < box_num; ++i) {

    if (merged[i]) continue;
    std::vector<UltraBox> buf;

    buf.push_back(input[i]);
    merged[i] = 1;

    float h0 = input[i].y2 - input[i].y1 + 1.0f;
    float w0 = input[i].x2 - input[i].x1 + 1.0f;

    float area0 = h0 * w0;

    for (unsigned int j = i + 1; j < box_num; ++j) {
      if (merged[j]) continue;

      float inner_x0 = input[i].x1 > input[j].x1 ? input[i].x1 : input[j].x1;
      float inner_y0 = input[i].y1 > input[j].y1 ? input[i].y1 : input[j].y1;

      float inner_x1 = input[i].x2 < input[j].x2 ? input[i].x2 : input[j].x2;
      float inner_y1 = input[i].y2 < input[j].y2 ? input[i].y2 : input[j].y2;

      float inner_h = inner_y1 - inner_y0 + 1.0f;
      float inner_w = inner_x1 - inner_x0 + 1.0f;

      if (inner_h <= 0.f || inner_w <= 0.f)
        continue;

      float inner_area = inner_h * inner_w;

      float h1 = input[j].y2 - input[j].y1 + 1.0f;
      float w1 = input[j].x2 - input[j].x1 + 1.0f;

      float area1 = h1 * w1;

      float iou = inner_area / (area0 + area1 - inner_area);

      if (iou > iou_threshold) {
        merged[j] = 1;
        buf.push_back(input[j]);
      }

    }
    output.push_back(buf[0]);

    // keep top k
    count += 1;
    if (count >= topk)
      break;
  }
}

cv::Mat UltraFace::draw_boxes(const cv::Mat &mat, const std::vector<UltraBox> &boxes) {
  cv::Mat canva = mat.clone();
  if (boxes.empty()) return canva;

  for (const auto &box: boxes) {
    int x1 = static_cast<int>(box.x1);
    int y1 = static_cast<int>(box.y1);
    int x2 = static_cast<int>(box.x2);
    int y2 = static_cast<int>(box.y2);
    int w = x2 - x1 + 1;
    int h = y2 - y1 + 1;
    cv::rectangle(canva, cv::Rect(x1, y1, w, h), cv::Scalar(255, 255, 0), 2);
  }
  return canva;
}

void UltraFace::draw_boxes_inplace(cv::Mat &mat_inplace, const std::vector<UltraBox> &boxes) {
  if (boxes.empty()) return;

  for (const auto &box: boxes) {
    int x1 = static_cast<int>(box.x1);
    int y1 = static_cast<int>(box.y1);
    int x2 = static_cast<int>(box.x2);
    int y2 = static_cast<int>(box.y2);
    int w = x2 - x1 + 1;
    int h = y2 - y1 + 1;
    cv::rectangle(mat_inplace, cv::Rect(x1, y1, w, h), cv::Scalar(255, 255, 0), 2);
  }
}


























