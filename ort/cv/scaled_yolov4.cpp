//
// Created by DefTruth on 2021/8/8.
//

#include "scaled_yolov4.h"
#include "ort/core/ort_utils.h"
#include "lite/utils.h"

using ortcv::ScaledYoloV4;

Ort::Value ScaledYoloV4::transform(const cv::Mat &mat)
{
  cv::Mat canva = mat.clone();
  cv::cvtColor(canva, canva, cv::COLOR_BGR2RGB);
  cv::resize(canva, canva, cv::Size(input_node_dims.at(3),
                                    input_node_dims.at(2)));
  // (1,3,320|640|1280,320|640|1280) 1xCXHXW
  ortcv::utils::transform::normalize_inplace(canva, mean_val, scale_val); // float32
  return ortcv::utils::transform::create_tensor(
      canva, input_node_dims, memory_info_handler,
      input_values_handler, ortcv::utils::transform::CHW);
}


void ScaledYoloV4::detect(const cv::Mat &mat, std::vector<types::Boxf> &detected_boxes,
                          float score_threshold, float iou_threshold, unsigned int topk,
                          unsigned int nms_type)
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

void ScaledYoloV4::generate_bboxes(std::vector<types::Boxf> &bbox_collection,
                                   std::vector<Ort::Value> &output_tensors,
                                   float score_threshold, float img_height,
                                   float img_width)
{

  Ort::Value &pred = output_tensors.at(0); // (1,n,85=5+80=cxcy+cwch+obj_conf+cls_conf)
  auto pred_dims = output_node_dims.at(0); // (1,n,85)
  const unsigned int num_anchors = pred_dims.at(1); // n = ?
  const unsigned int num_classes = pred_dims.at(2) - 5;
  const float input_height = static_cast<float>(input_node_dims.at(2)); // e.g 640
  const float input_width = static_cast<float>(input_node_dims.at(3)); // e.g 640
  const float scale_height = img_height / input_height;
  const float scale_width = img_width / input_width;

  bbox_collection.clear();
  unsigned int count = 0;
  for (unsigned int i = 0; i < num_anchors; ++i)
  {
    float obj_conf = pred.At<float>({0, i, 4});
    if (obj_conf < score_threshold) continue; // filter first.

    float cls_conf = pred.At<float>({0, i, 5});
    unsigned int label = 0;
    for (unsigned int j = 0; j < num_classes; ++j)
    {
      float tmp_conf = pred.At<float>({0, i, j + 5});
      if (tmp_conf > cls_conf)
      {
        cls_conf = tmp_conf;
        label = j;
      }
    }
    float conf = obj_conf * cls_conf; // cls_conf (0.,1.)
    if (conf < score_threshold) continue; // filter

    float cx = pred.At<float>({0, i, 0});
    float cy = pred.At<float>({0, i, 1});
    float w = pred.At<float>({0, i, 2});
    float h = pred.At<float>({0, i, 3});

    types::Boxf box;
    box.x1 = (cx - w / 2.f) * scale_width;
    box.y1 = (cy - h / 2.f) * scale_height;
    box.x2 = (cx + w / 2.f) * scale_width;
    box.y2 = (cy + h / 2.f) * scale_height;
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

void ScaledYoloV4::nms(std::vector<types::Boxf> &input, std::vector<types::Boxf> &output,
                       float iou_threshold, unsigned int topk, unsigned int nms_type)
{
  if (nms_type == NMS::BLEND) lite::utils::blending_nms(input, output, iou_threshold, topk);
  else if (nms_type == NMS::OFFSET) lite::utils::offset_nms(input, output, iou_threshold, topk);
  else lite::utils::hard_nms(input, output, iou_threshold, topk);
}
