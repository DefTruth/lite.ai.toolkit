//
// Created by DefTruth on 2021/6/5.
//

#include "ssd.h"
#include "ort/core/ort_utils.h"
#include "lite/utils.h"

using ortcv::SSD;

Ort::Value SSD::transform(const cv::Mat &mat)
{
  cv::Mat canva = mat.clone();
  cv::cvtColor(canva, canva, cv::COLOR_BGR2RGB);
  cv::resize(canva, canva, cv::Size(input_node_dims.at(3),
                                    input_node_dims.at(2))); // (1200,1200,3)

  canva.convertTo(canva, CV_32FC3, 1.0f / 255.0f, 0.f); // (0.,1.)
  ortcv::utils::transform::normalize_inplace(canva, mean_vals, scale_vals); // float32
  return ortcv::utils::transform::create_tensor(
      canva, input_node_dims, memory_info_handler,
      input_values_handler, ortcv::utils::transform::CHW);
}

void SSD::detect(const cv::Mat &mat, std::vector<types::Boxf> &detected_boxes,
                 float score_threshold, float iou_threshold,
                 unsigned int topk, unsigned int nms_type)
{
  if (mat.empty()) return;
  // this->transform(mat);
  float img_height = static_cast<float>(mat.rows);
  float img_width = static_cast<float>(mat.cols);

  // 1. make input tensor
  Ort::Value input_tensor = this->transform(mat);
  // 2. inference scores & boxes.
  auto output_tensors = ort_session->Run(
      Ort::RunOptions{nullptr}, input_node_names.data(),
      &input_tensor, 1, output_node_names.data(), num_outputs
  );
  // 3. rescale & exclude.
  std::vector<types::Boxf> bbox_collection;
  this->generate_bboxes(bbox_collection, output_tensors, score_threshold, img_height, img_width);
  // 4. hard|blend nms with topk.
  this->nms(bbox_collection, detected_boxes, iou_threshold, topk, nms_type);
}

void SSD::generate_bboxes(std::vector<types::Boxf> &bbox_collection,
                          std::vector<Ort::Value> &output_tensors,
                          float score_threshold, float img_height,
                          float img_width)
{
  Ort::Value &bboxes = output_tensors.at(0); // (1,n,4)
  Ort::Value &labels = output_tensors.at(1); // (1,n) bg+cls=1+80
  Ort::Value &scores = output_tensors.at(2); // (1,n) n is dynamic
  auto bboxes_dims = bboxes.GetTypeInfo().GetTensorTypeAndShapeInfo().GetShape();
  const unsigned int num_anchors = bboxes_dims.at(1);

  bbox_collection.clear();
  unsigned int count = 0;
  for (unsigned int i = 0; i < num_anchors; ++i)
  {
    float conf = scores.At<float>({0, i});
    if (conf < score_threshold) continue; // filter
    unsigned int label = labels.At<unsigned int>({0, i}) - 1;

    types::Boxf box;
    box.x1 = bboxes.At<float>({0, i, 0}) * (float) img_width;
    box.y1 = bboxes.At<float>({0, i, 1}) * (float) img_height;
    box.x2 = bboxes.At<float>({0, i, 2}) * (float) img_width;
    box.y2 = bboxes.At<float>({0, i, 3}) * (float) img_height;
    box.score = conf;
    box.label = label;
    box.label_text = class_names[label];
    box.flag = true;
    bbox_collection.push_back(box);

    count += 1; // limit boxes for nms.
    if (count > max_nms)
      break;
  }

#if LITEORT_DEBUG
  std::cout << "detected num_anchors: " << num_anchors << "\n";
  std::cout << "generate_bboxes num: " << bbox_collection.size() << "\n";
#endif

}

void SSD::nms(std::vector<types::Boxf> &input, std::vector<types::Boxf> &output,
              float iou_threshold, unsigned int topk, unsigned int nms_type)
{
  if (nms_type == NMS::BLEND) lite::utils::blending_nms(input, output, iou_threshold, topk);
  else if (nms_type == NMS::OFFSET) lite::utils::offset_nms(input, output, iou_threshold, topk);
  else lite::utils::hard_nms(input, output, iou_threshold, topk);
}