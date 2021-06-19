//
// Created by DefTruth on 2021/5/23.
//

#include "yolov3.h"
#include "ort/core/ort_utils.h"

using ortcv::YoloV3;

// yolov3 is an multi-inputs & multi-outputs & dynamic shape
// (dynamic: batch,input_height,input_width)
YoloV3::YoloV3(const std::string &_onnx_path, unsigned int _num_threads) :
    log_id(_onnx_path.data()), num_threads(_num_threads)
{
#ifdef LITEHUB_WIN32
  std::wstring _w_onnx_path(ortcv::utils::to_wstring(_onnx_path));
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
  ort_session = new Ort::Session(ort_env, onnx_path, session_options);

  Ort::AllocatorWithDefaultOptions allocator;
  // 2. input name & input dims
  num_inputs = ort_session->GetInputCount();
  input_node_names.resize(num_inputs);
  // 3. initial input node dims.
  input_node_dims.push_back({batch_size, 3, input_height, input_width}); // input_1 dims
  input_node_dims.push_back({batch_size, 2}); // image_shape dims
  input_tensor_sizes.push_back(batch_size * 3 * input_height * input_width);
  input_tensor_sizes.push_back(batch_size * 2);
  input_1_values_handler.resize(batch_size * 3 * input_height * input_width);
  image_shape_values_handler.resize(batch_size * 2);
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

YoloV3::~YoloV3()
{
  if (ort_session)
    delete ort_session;
  ort_session = nullptr;
}

void YoloV3::print_debug_string()
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

std::vector<Ort::Value> YoloV3::transform(const std::vector<cv::Mat> &mats)
{
  cv::Mat canvas = mats.at(0).clone();  // (h,w,3) uint8 mats contains one mat only.
  // multi inputs: input_1 image_shape
  // input_1 with shape (1,3,416,416);
  // image_shape is original shape of source image.
  std::vector<int64_t> input_1_dims = input_node_dims.at(0); // (1,3,416,416); reference
  std::vector<int64_t> image_shape_dims = input_node_dims.at(1); // (1,2);

  const unsigned int image_height = canvas.rows;
  const unsigned int image_width = canvas.cols;

  const float scale = std::fmin(
      (float) input_width / (float) image_width,
      (float) input_height / (float) image_height
  );

  const unsigned int nw = static_cast<unsigned int>((float) image_width * scale);
  const unsigned int nh = static_cast<unsigned int>((float) image_height * scale);

  cv::resize(canvas, canvas, cv::Size(nw, nh));
  cv::cvtColor(canvas, canvas, cv::COLOR_BGR2RGB);

  cv::Mat canvas_pad(input_height, input_width, CV_8UC3, 128);
  const unsigned int x1 = (input_width - nw) / 2;
  const unsigned int y1 = (input_height - nh) / 2;
  cv::Rect roi(x1, y1, nw, nh);
  canvas.convertTo(canvas_pad(roi), CV_8UC3); // padding

  std::vector<Ort::Value> input_tensors;
  // make tensor of input_1 & image_shape
  ortcv::utils::transform::normalize_inplace(canvas_pad, mean_val, scale_val); // float32 (0.,1.)
  input_tensors.emplace_back(ortcv::utils::transform::create_tensor(
      canvas_pad, input_1_dims, memory_info_handler,
      input_1_values_handler, ortcv::utils::transform::CHW
  )); // input_1
  image_shape_values_handler[0] = static_cast<float>(image_height);
  image_shape_values_handler[1] = static_cast<float>(image_width);
  input_tensors.emplace_back(Ort::Value::CreateTensor<float>(
      memory_info_handler, image_shape_values_handler.data(),
      input_tensor_sizes.at(1), image_shape_dims.data(),
      image_shape_dims.size())
  ); // image_shape

  return input_tensors;
}

void YoloV3::detect(const cv::Mat &mat, std::vector<types::Boxf> &detected_boxes)
{
  if (mat.empty()) return;
  std::vector<cv::Mat> mats;
  mats.push_back(mat);
  // 1. make input tensor
  std::vector<Ort::Value> input_tensors = this->transform(mats);
  // 2. inference boxes & scores & indices.

  auto output_tensors = ort_session->Run(
      Ort::RunOptions{nullptr}, input_node_names.data(),
      input_tensors.data(), num_inputs, output_node_names.data(),
      num_outputs
  );
  /** 3. generate final detected bounding boxes.
   * boxes: (1x'n_candidates'x4) the coordinates of all anchor boxes (y1,x1,y2,x2)
   * scores: (1x80x'n_candidates') the scores of all anchor boxes per class
   * indices: ('nbox'x3) selected indices from the boxes tensor after NMS.
   * The selected index format is (batch_index, class_index, box_index)
   * */
  this->generate_bboxes(detected_boxes, output_tensors);

}


void YoloV3::generate_bboxes(std::vector<types::Boxf> &bbox_collection,
                             std::vector<Ort::Value> &output_tensors)
{
  Ort::Value &boxes = output_tensors.at(0); // (1,'num_anchors',4) (1, 10647, 4)
  Ort::Value &scores = output_tensors.at(1); // (1,80,'num_anchors') (1, 80, 10647)
  Ort::Value &indices = output_tensors.at(2); // (num_selected,3)
  auto indices_dims = indices.GetTypeInfo().GetTensorTypeAndShapeInfo().GetShape();

  const unsigned int num_selected = indices_dims.at(0);

  bbox_collection.clear();
  for (unsigned int i = 0; i < num_selected; ++i)
  {
    unsigned int batch_index = indices.At<unsigned int>({i, 0});
    unsigned int class_index = indices.At<unsigned int>({i, 1});
    unsigned int box_index = indices.At<unsigned int>({i, 2});

    types::Boxf box;
    box.x1 = boxes.At<float>({batch_index, box_index, 1});
    box.y1 = boxes.At<float>({batch_index, box_index, 0});
    box.x2 = boxes.At<float>({batch_index, box_index, 3});
    box.y2 = boxes.At<float>({batch_index, box_index, 2});
    box.score = scores.At<float>({batch_index, class_index, box_index});
    box.label = class_index;
    box.label_text = class_names[class_index];
    box.flag = true;
    bbox_collection.push_back(box);
  }
#if LITEORT_DEBUG
  auto boxes_dims = boxes.GetTypeInfo().GetTensorTypeAndShapeInfo().GetShape();
  const unsigned int num_anchors = boxes_dims.at(1);
  std::cout << "detected num_anchors: " << num_anchors << "\n";
  std::cout << "generate_bboxes num: " << bbox_collection.size() << "\n";
#endif
}


































