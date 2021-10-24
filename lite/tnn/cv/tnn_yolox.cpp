//
// Created by DefTruth on 2021/10/17.
//

#include "tnn_yolox.h"
#include "lite/utils.h"

using tnncv::TNNYoloX;

TNNYoloX::TNNYoloX(const std::string &_proto_path,
                   const std::string &_model_path,
                   unsigned int _num_threads) :
    BasicTNNHandler(_proto_path, _model_path, _num_threads)

{
}

void TNNYoloX::transform(const cv::Mat &mat)
{
  cv::Mat canvas = mat.clone();
  cv::resize(canvas, canvas, cv::Size(input_width, input_height));
  cv::cvtColor(canvas, canvas, cv::COLOR_BGR2RGB);
  // push into input_mat
  input_mat = std::make_shared<tnn::Mat>(input_device_type, tnn::N8UC3,
                                         input_shape, (void *)canvas.data);
  if (!input_mat->GetData())
  {
#ifdef LITETNN_DEBUG
    std::cout << "input_mat == nullptr! transform failed\n";
#endif
  }
}

void TNNYoloX::detect(const cv::Mat &mat, std::vector<types::Boxf> &detected_boxes,
                      float score_threshold, float iou_threshold,
                      unsigned int topk, unsigned int nms_type)
{
  if (mat.empty()) return;
  float img_height = static_cast<float>(mat.rows);
  float img_width = static_cast<float>(mat.cols);

  // 1. make input tensor
  this->transform(mat);
  // 2. set input_mat
  tnn::MatConvertParam input_cvt_param;
  input_cvt_param.scale = scale_vals;
  input_cvt_param.bias = bias_vals;

  tnn::Status status;
  status = instance->SetInputMat(input_mat, input_cvt_param);
  if (status != tnn::TNN_OK)
  {
#ifdef LITETNN_DEBUG
    std::cout << "instance->SetInputMat failed!:"
              << status.description().c_str() << "\n";
#endif
    return;
  }

  // 3. forward
  status = instance->Forward();
  if (status != tnn::TNN_OK)
  {
#ifdef LITETNN_DEBUG
    std::cout << "instance->Forward failed!:"
              << status.description().c_str() << "\n";
#endif
    return;
  }
  // 4. fetch output mat
  std::shared_ptr<tnn::Mat> pred_mat;
  tnn::MatConvertParam pred_cvt_param; // default

  status = instance->GetOutputMat(pred_mat, pred_cvt_param, "outputs", output_device_type);
  if (status != tnn::TNN_OK)
  {
#ifdef LITETNN_DEBUG
    std::cout << "instance->GetOutputMat failed!:"
              << status.description().c_str() << "\n";
#endif
    return;
  }
  // 5. rescale & exclude.
  std::vector<types::Boxf> bbox_collection;
  this->generate_bboxes(bbox_collection, pred_mat, score_threshold, img_height, img_width);
  // 6. hard|blend|offset nms with topk.
  this->nms(bbox_collection, detected_boxes, iou_threshold, topk, nms_type);
}

void TNNYoloX::generate_anchors(const int target_height,
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

void TNNYoloX::generate_bboxes(std::vector<types::Boxf> &bbox_collection,
                               const std::shared_ptr<tnn::Mat> &pred_mat,
                               float score_threshold, float img_height,
                               float img_width)
{
  auto pred_dims = pred_mat->GetDims();
  const unsigned int num_anchors = pred_dims.at(1); // n = ?
  const unsigned int num_classes = pred_dims.at(2) - 5;
  const float scale_height = img_height / (float) input_height;
  const float scale_width = img_width / (float) input_width;

  std::vector<YoloXAnchor> anchors;
  std::vector<int> strides = {8, 16, 32}; // might have stride=64
  this->generate_anchors(input_height, input_width, strides, anchors);

  bbox_collection.clear();
  unsigned int count = 0;
  for (unsigned int i = 0; i < num_anchors; ++i)
  {
    const float *offset_obj_cls_ptr =
        (float *) pred_mat->GetData() + (i * (num_classes + 5));
    float obj_conf = offset_obj_cls_ptr[4];
    if (obj_conf < score_threshold) continue; // filter first.

    float cls_conf = offset_obj_cls_ptr[5];
    unsigned int label = 0;
    for (unsigned int j = 0; j < num_classes; ++j)
    {
      float tmp_conf = offset_obj_cls_ptr[j + 5];
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

    float dx = offset_obj_cls_ptr[0];
    float dy = offset_obj_cls_ptr[1];
    float dw = offset_obj_cls_ptr[2];
    float dh = offset_obj_cls_ptr[3];

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
#if LITETNN_DEBUG
  std::cout << "detected num_anchors: " << num_anchors << "\n";
  std::cout << "generate_bboxes num: " << bbox_collection.size() << "\n";
#endif
}

void TNNYoloX::nms(std::vector<types::Boxf> &input, std::vector<types::Boxf> &output,
                   float iou_threshold, unsigned int topk,
                   unsigned int nms_type)
{
  if (nms_type == NMS::BLEND) lite::utils::blending_nms(input, output, iou_threshold, topk);
  else if (nms_type == NMS::OFFSET) lite::utils::offset_nms(input, output, iou_threshold, topk);
  else lite::utils::hard_nms(input, output, iou_threshold, topk);
}




















