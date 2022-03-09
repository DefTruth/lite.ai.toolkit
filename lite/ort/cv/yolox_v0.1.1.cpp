//
// Created by DefTruth on 2021/11/6.
//

#include "yolox_v0.1.1.h"
#include "lite/ort/core/ort_utils.h"
#include "lite/utils.h"

using ortcv::YoloX_V_0_1_1;

Ort::Value YoloX_V_0_1_1::transform(const cv::Mat &mat_rs)
{
  cv::Mat canvas = mat_rs.clone();
  // There is no normalization for the latest official C++ implementation of
  // v0.1.1 YOLOX model using ncnn. Reference:
  // [1] https://github.com/Megvii-BaseDetection/YOLOX/blob/main/demo/ncnn/cpp/yolox.cpp
  return ortcv::utils::transform::create_tensor(
      canvas, input_node_dims, memory_info_handler,
      input_values_handler, ortcv::utils::transform::CHW);
}

void YoloX_V_0_1_1::resize_unscale(const cv::Mat &mat, cv::Mat &mat_rs,
                                   int target_height, int target_width,
                                   YoloXScaleParams &scale_params)
{
  if (mat.empty()) return;
  int img_height = static_cast<int>(mat.rows);
  int img_width = static_cast<int>(mat.cols);

  mat_rs = cv::Mat(target_height, target_width, CV_8UC3,
                   cv::Scalar(114, 114, 114));
  // scale ratio (new / old) new_shape(h,w)
  float w_r = (float) target_width / (float) img_width;
  float h_r = (float) target_height / (float) img_height;
  float r = std::min(w_r, h_r);
  // compute padding
  int new_unpad_w = static_cast<int>((float) img_width * r); // floor
  int new_unpad_h = static_cast<int>((float) img_height * r); // floor
  int pad_w = target_width - new_unpad_w; // >=0
  int pad_h = target_height - new_unpad_h; // >=0

  int dw = pad_w / 2;
  int dh = pad_h / 2;

  // resize with unscaling
  cv::Mat new_unpad_mat = mat.clone();
  cv::resize(new_unpad_mat, new_unpad_mat, cv::Size(new_unpad_w, new_unpad_h));
  new_unpad_mat.copyTo(mat_rs(cv::Rect(dw, dh, new_unpad_w, new_unpad_h)));

  // record scale params.
  scale_params.r = r;
  scale_params.dw = dw;
  scale_params.dh = dh;
  scale_params.new_unpad_w = new_unpad_w;
  scale_params.new_unpad_h = new_unpad_h;
  scale_params.flag = true;
}

void YoloX_V_0_1_1::detect(const cv::Mat &mat, std::vector<types::Boxf> &detected_boxes,
                           float score_threshold, float iou_threshold,
                           unsigned int topk, unsigned int nms_type)
{
  if (mat.empty()) return;
  const int input_height = input_node_dims.at(2);
  const int input_width = input_node_dims.at(3);
  int img_height = static_cast<int>(mat.rows);
  int img_width = static_cast<int>(mat.cols);

  // resize & unscale
  cv::Mat mat_rs;
  YoloXScaleParams scale_params;
  this->resize_unscale(mat, mat_rs, input_height, input_width, scale_params);

  // 1. make input tensor
  Ort::Value input_tensor = this->transform(mat_rs);
  // 2. inference scores & boxes.
  auto output_tensors = ort_session->Run(
      Ort::RunOptions{nullptr}, input_node_names.data(),
      &input_tensor, 1, output_node_names.data(), num_outputs
  );
  // 3. rescale & exclude.
  std::vector<types::Boxf> bbox_collection;
  this->generate_bboxes(scale_params, bbox_collection, output_tensors, score_threshold, img_height, img_width);
  // 4. hard|blend|offset nms with topk.
  this->nms(bbox_collection, detected_boxes, iou_threshold, topk, nms_type);
}

// Issue: https://github.com/DefTruth/lite.ai/issues/9
// Note!!!: The implementation of Anchor generation in Lite.AI is slightly different
// with the official one in order to fix the inference error for non-square input shape.
// Official: https://github.com/Megvii-BaseDetection/YOLOX/blob/main/demo/ncnn/cpp/yolox.cpp
/** Official implementation. It assumes that the input shape must be a square.
 *  When you use the YOLOX model that was trained by yourself, but the input tensor of
 *  the model is not square, you will encounter an error. So, I decided to extend the
 *  official implementation for compatibility with square and non-square input.
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

void YoloX_V_0_1_1::generate_anchors(const int target_height,
                                     const int target_width,
                                     std::vector<int> &strides,
                                     std::vector<YoloXAnchor> &anchors)
{
  for (auto stride : strides)
  {
    int num_grid_w = target_width / stride;
    int num_grid_h = target_height / stride;
    for (int g1 = 0; g1 < num_grid_h; ++g1)
    {
      for (int g0 = 0; g0 < num_grid_w; ++g0)
      {
#ifdef LITE_WIN32
        YoloXAnchor anchor;
        anchor.grid0 = g0;
        anchor.grid1 = g1;
        anchor.stride = stride;
        anchors.push_back(anchor);
#else
        anchors.push_back((YoloXAnchor) {g0, g1, stride});
#endif
      }
    }
  }
}


void YoloX_V_0_1_1::generate_bboxes(const YoloXScaleParams &scale_params,
                                    std::vector<types::Boxf> &bbox_collection,
                                    std::vector<Ort::Value> &output_tensors,
                                    float score_threshold, int img_height,
                                    int img_width)
{
  Ort::Value &pred = output_tensors.at(0); // (1,n,85=5+80=cxcy+cwch+obj_conf+cls_conf)
  auto pred_dims = output_node_dims.at(0); // (1,n,85)
  const unsigned int num_anchors = pred_dims.at(1); // n = ?
  const unsigned int num_classes = pred_dims.at(2) - 5;
  const float input_height = static_cast<float>(input_node_dims.at(2)); // e.g 640
  const float input_width = static_cast<float>(input_node_dims.at(3)); // e.g 640

  std::vector<YoloXAnchor> anchors;
  std::vector<int> strides = {8, 16, 32}; // might have stride=64
  this->generate_anchors(input_height, input_width, strides, anchors);

  float r_ = scale_params.r;
  int dw_ = scale_params.dw;
  int dh_ = scale_params.dh;

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
    float w = std::exp(dw) * (float) stride;
    float h = std::exp(dh) * (float) stride;
    float x1 = ((cx - w / 2.f) - (float) dw_) / r_;
    float y1 = ((cy - h / 2.f) - (float) dh_) / r_;
    float x2 = ((cx + w / 2.f) - (float) dw_) / r_;
    float y2 = ((cy + h / 2.f) - (float) dh_) / r_;

    types::Boxf box;
    box.x1 = std::max(0.f, x1);
    box.y1 = std::max(0.f, y1);
    box.x2 = std::min(x2, (float) img_width);
    box.y2 = std::min(y2, (float) img_height);
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


void YoloX_V_0_1_1::nms(std::vector<types::Boxf> &input, std::vector<types::Boxf> &output,
                        float iou_threshold, unsigned int topk, unsigned int nms_type)
{
  if (nms_type == NMS::BLEND) lite::utils::blending_nms(input, output, iou_threshold, topk);
  else if (nms_type == NMS::OFFSET) lite::utils::offset_nms(input, output, iou_threshold, topk);
  else lite::utils::hard_nms(input, output, iou_threshold, topk);
}


