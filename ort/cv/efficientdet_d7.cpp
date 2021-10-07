//
// Created by DefTruth on 2021/8/15.
//

#include "efficientdet_d7.h"
#include "ort/core/ort_utils.h"
#include "lite/utils.h"

using ortcv::EfficientDetD7;

Ort::Value EfficientDetD7::transform(const cv::Mat &mat)
{
  cv::Mat canva = mat.clone();
  cv::cvtColor(canva, canva, cv::COLOR_BGR2RGB);
  // resize without padding, todo: add padding as the official Python implementation.
  cv::resize(canva, canva, cv::Size(input_node_dims.at(3),
                                    input_node_dims.at(2)));
  // (1,3,1536,1536) 1xCXHXW
  canva.convertTo(canva, CV_32FC3, 1.0f / 255.f, 0.f);
  ortcv::utils::transform::normalize_inplace(canva, mean_vals, scale_vals); // float32
  return ortcv::utils::transform::create_tensor(
      canva, input_node_dims, memory_info_handler,
      input_values_handler, ortcv::utils::transform::CHW);
}

void EfficientDetD7::detect(const cv::Mat &mat, std::vector<types::Boxf> &detected_boxes,
                            float score_threshold, float iou_threshold,
                            unsigned int topk, unsigned int nms_type)
{
  if (mat.empty()) return;
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
  // 4. hard|blend|offset nms with topk.
  this->nms(bbox_collection, detected_boxes, iou_threshold, topk, nms_type);
}

// ref: https://github.com/zylo117/Yet-Another-EfficientDet-Pytorch/blob/master/efficientdet/utils.py
void EfficientDetD7::generate_anchors(const float target_height, const float target_width)
{
  if (!anchors_buffer.empty()) return;

  // generate once.
  for (const auto &stride: strides)
  {
    // create grid with a specific stride. Under a specific stride,
    // 9 Anchors of the same anchor point are stacked together in order
    for (float yv = stride / 2.0f; yv < target_height; yv += stride)
    {
      for (float xv = stride / 2.0f; xv < target_width; xv += stride)
      {
        for (const auto &scale: scales)
        {
          for (const auto &ratio: ratios)
          {
            float base_anchor_size = anchor_scale * stride * scale;
            // aw/2 and ah/2, according to input size with different ratio.
            float anchor_size_x_2 = base_anchor_size * ratio[0] / 2.0f;
            float anchor_size_y_2 = base_anchor_size * ratio[1] / 2.0f;

            float y1 = yv - anchor_size_y_2; // cy - ah/2
            float x1 = xv - anchor_size_x_2; // cx - aw/2
            float y2 = yv + anchor_size_y_2; // cy + ah/2
            float x2 = xv + anchor_size_x_2; // cx + aw/2
#ifdef LITE_WIN32
            EfficientDetD7Anchor anchor;
            anchor.y1 = y1;
            anchor.x1 = x1;
            anchor.y2 = y2;
            anchor.x2 = x2;
            anchors_buffer.push_back(anchor);
#else
            anchors_buffer.push_back((EfficientDetD7Anchor) {y1, x1, y2, x2});
#endif
          } // end ratios 3
        } // end scale 3
      }
    } // end grid
  } // end strides
}


void EfficientDetD7::generate_bboxes(std::vector<types::Boxf> &bbox_collection,
                                     std::vector<Ort::Value> &output_tensors,
                                     float score_threshold,
                                     float img_height, float img_width)
{
  Ort::Value &regression = output_tensors.at(0); // (1,n,4) (dy, dx, dh, dw)]
  Ort::Value &classification = output_tensors.at(1); // (1,n,90) 90 classes
  auto reg_dims = output_node_dims.at(0); // (1,n,4)
  auto cls_dims = output_node_dims.at(1); // (1,n,90)
  const unsigned int num_anchors = reg_dims.at(1); // n = ?
  const unsigned int num_classes = cls_dims.at(2); // 90
  const float input_height = static_cast<float>(input_node_dims.at(2)); // e.g 512
  const float input_width = static_cast<float>(input_node_dims.at(3)); // e.g 512
  const float scale_height = img_height / input_height;
  const float scale_width = img_width / input_width;

  this->generate_anchors(input_height, input_width); // once
  if (anchors_buffer.size() != num_anchors)
    throw std::runtime_error("mismatch size for anchors_buffer and num_anchor.");

  bbox_collection.clear();
  unsigned int count = 0;
  for (unsigned int i = 0; i < num_anchors; ++i)
  {
    float cls_conf = classification.At<float>({0, i, 0});
    unsigned int label = 0;
    for (unsigned int j = 0; j < num_classes; ++j)
    {
      float tmp_conf = classification.At<float>({0, i, j});
      if (tmp_conf > cls_conf)
      {
        cls_conf = tmp_conf;
        label = j;
      }
    } // argmax
    if (cls_conf < score_threshold) continue; // filter

    float ay1 = anchors_buffer.at(i).y1;
    float ax1 = anchors_buffer.at(i).x1;
    float ay2 = anchors_buffer.at(i).y2;
    float ax2 = anchors_buffer.at(i).x2;
    float cya = (ay1 + ay2) / 2.0f; // center
    float cxa = (ax1 + ax2) / 2.0f;
    float ha = ay2 - ay1;
    float wa = ax2 - ax1;

    float dy = regression.At<float>({0, i, 0});
    float dx = regression.At<float>({0, i, 1});
    float dh = regression.At<float>({0, i, 2});
    float dw = regression.At<float>({0, i, 3});

    float cx = dx * wa + cxa;
    float cy = dy * ha + cya;
    float w = std::expf(dw) * wa;
    float h = std::expf(dh) * ha;

    types::Boxf box;
    box.x1 = (cx - w / 2.f) * scale_width;
    box.y1 = (cy - h / 2.f) * scale_height;
    box.x2 = (cx + w / 2.f) * scale_width;
    box.y2 = (cy + h / 2.f) * scale_height;
    box.score = cls_conf;
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


void EfficientDetD7::nms(std::vector<types::Boxf> &input, std::vector<types::Boxf> &output,
                         float iou_threshold, unsigned int topk, unsigned int nms_type)
{
  if (nms_type == NMS::BLEND) lite::utils::blending_nms(input, output, iou_threshold, topk);
  else if (nms_type == NMS::OFFSET) lite::utils::offset_nms(input, output, iou_threshold, topk);
  else lite::utils::hard_nms(input, output, iou_threshold, topk);
}
