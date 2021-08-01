//
// Created by DefTruth on 2021/7/20.
//

#include "yolox.h"
#include "ort/core/ort_utils.h"

using ortcv::YoloX;

Ort::Value YoloX::transform(const cv::Mat &mat)
{
  cv::Mat canva = mat.clone();
  cv::cvtColor(canva, canva, cv::COLOR_BGR2RGB);
  // resize without padding, todo: add padding as the official Python implementation.
  cv::resize(canva, canva, cv::Size(input_node_dims.at(3),
                                    input_node_dims.at(2)));
  // (1,3,640,640) 1xCXHXW

  ortcv::utils::transform::normalize_inplace(canva, mean_vals, scale_vals); // float32
  return ortcv::utils::transform::create_tensor(
      canva, input_node_dims, memory_info_handler,
      input_values_handler, ortcv::utils::transform::CHW);
}

void YoloX::detect(const cv::Mat &mat, std::vector<types::Boxf> &detected_boxes,
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

// Issue: https://github.com/DefTruth/lite.ai/issues/9
// Note!!!: The implementation of Anchor generation in Lite.AI is slightly different
// with the official one in order to fix the inference error for non-square input shape.
// Official: https://github.com/Megvii-BaseDetection/YOLOX/blob/main/demo/ncnn/cpp/yolox.cpp
/** Official implementation. It assumes that the input shape must be a square.
 *  When you use the YOLOX you trained by yourself, but the input tensor of the model
 *  is not square, you will encounter an error. I decided to extend the official
 *  implementation for compatibility with square and non-square input.
 *
 * static void generate_grids_and_stride(const int target_size, std::vector<int>& strides,
 *                                       std::vector<GridAndStride>& grid_strides)
 * {
 *  for (auto stride : strides)
 *  {
 *       int num_grid = target_size / stride;
 *       for (int g1 = 0; g1 < num_grid; g1++)
 *       {
 *           for (int g0 = 0; g0 < num_grid; g0++)
 *           {
 *               grid_strides.push_back((GridAndStride){g0, g1, stride});
 *           }
 *       }
 *   }
 * }
 */

void YoloX::generate_anchors(const int target_height,
                             const int target_width,
                             std::vector<int> &strides,
                             std::vector<Anchor> &anchors)
{
  for (auto stride : strides)
  {
    int num_grid_w = target_width / stride;
    int num_grid_h = target_height / stride;
    for (int g1 = 0; g1 < num_grid_h; g1++)
    {
      for (int g0 = 0; g0 < num_grid_w; g0++)
      {
        anchors.push_back((Anchor) {g0, g1, stride});
      }
    }
  }
}


void YoloX::generate_bboxes(std::vector<types::Boxf> &bbox_collection,
                            std::vector<Ort::Value> &output_tensors,
                            float score_threshold,
                            float img_height, float img_width)
{
  Ort::Value &pred = output_tensors.at(0); // (1,n,85=5+80=cxcy+cwch+obj_conf+cls_conf)
  auto pred_dims = output_node_dims.at(0); // (1,n,85)
  const unsigned int num_anchors = pred_dims.at(1); // n = ?
  const unsigned int num_classes = pred_dims.at(2) - 5;
  const float input_height = static_cast<float>(input_node_dims.at(2)); // e.g 640
  const float input_width = static_cast<float>(input_node_dims.at(3)); // e.g 640
  const float scale_height = img_height / input_height;
  const float scale_width = img_width / input_width;

  std::vector<Anchor> anchors;
  std::vector<int> strides = {8, 16, 32}; // might have stride=64
  this->generate_anchors(input_height, input_width, strides, anchors);

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
    } // argmax
    float conf = obj_conf * cls_conf; // cls_conf (0.,1.)
    if (conf < score_threshold) continue; // filter

    const int grid0 = anchors.at(i).grid0;
    const int grid1 = anchors.at(i).grid1;
    const int stride = anchors.at(i).stride;

    float dx = pred.At<float>({0, i, 0});
    float dy = pred.At<float>({0, i, 1});
    float dw = pred.At<float>({0, i, 2});
    float dh = pred.At<float>({0, i, 3});

    float cx = (dx + (float) grid0) * (float) stride;
    float cy = (dy + (float) grid1) * (float) stride;
    float w = std::expf(dw) * (float) stride;
    float h = std::expf(dh) * (float) stride;

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


void YoloX::nms(std::vector<types::Boxf> &input, std::vector<types::Boxf> &output,
                float iou_threshold, unsigned int topk, unsigned int nms_type)
{
  if (nms_type == NMS::BLEND) ortcv::utils::blending_nms(input, output, iou_threshold, topk);
  else if (nms_type == NMS::OFFSET) ortcv::utils::offset_nms(input, output, iou_threshold, topk);
  else ortcv::utils::hard_nms(input, output, iou_threshold, topk);
}




















































