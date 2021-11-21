//
// Created by DefTruth on 2021/11/20.
//

#include "tnn_ultraface.h"
#include "lite/utils.h"

using tnncv::TNNUltraFace;

TNNUltraFace::TNNUltraFace(const std::string &_proto_path,
                           const std::string &_model_path,
                           unsigned int _num_threads) :
    BasicTNNHandler(_proto_path, _model_path, _num_threads)
{
}

void TNNUltraFace::transform(const cv::Mat &mat)
{
  cv::Mat mat_rs;
  cv::resize(mat, mat_rs, cv::Size(input_width, input_height));
  cv::cvtColor(mat_rs, mat_rs, cv::COLOR_BGR2RGB);
  // push into input_mat
  input_mat = std::make_shared<tnn::Mat>(input_device_type, tnn::N8UC3,
                                         input_shape, (void *) mat_rs.data);
  if (!input_mat->GetData())
  {
#ifdef LITETNN_DEBUG
    std::cout << "input_mat == nullptr! transform failed\n";
#endif
  }
}

void TNNUltraFace::detect(const cv::Mat &mat, std::vector<types::Boxf> &detected_boxes,
                          float score_threshold, float iou_threshold,
                          unsigned int topk, unsigned int nms_type)
{
  if (mat.empty()) return;
  float img_height = static_cast<float>(mat.rows);
  float img_width = static_cast<float>(mat.cols);

  // 1. make input mat
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
    std::cout << status.description().c_str() << "\n";
#endif
    return;
  }
  // 3. forward
  status = instance->Forward();
  if (status != tnn::TNN_OK)
  {
#ifdef LITETNN_DEBUG
    std::cout << status.description().c_str() << "\n";
#endif
    return;
  }
  // 4. rescale & exclude.
  std::vector<types::Boxf> bbox_collection;
  this->generate_bboxes(bbox_collection, instance, score_threshold, img_height, img_width);
  // 5. hard|blend nms with topk.
  this->nms(bbox_collection, detected_boxes, iou_threshold, topk, nms_type);
}

void TNNUltraFace::generate_bboxes(std::vector<types::Boxf> &bbox_collection,
                                   std::shared_ptr<tnn::Instance> &_instance,
                                   float score_threshold, float img_height,
                                   float img_width)
{
  std::shared_ptr<tnn::Mat> boxes; // (1,n,4)
  std::shared_ptr<tnn::Mat> scores; // (1,n,2)
  tnn::MatConvertParam cvt_param;
  tnn::Status status_boxes;
  tnn::Status status_scores;

  status_boxes = _instance->GetOutputMat(boxes, cvt_param, "boxes", output_device_type);
  status_scores = _instance->GetOutputMat(scores, cvt_param, "scores", output_device_type);

  if (status_boxes != tnn::TNN_OK || status_scores != tnn::TNN_OK)
  {
#ifdef LITETNN_DEBUG
    std::cout << status_boxes.description().c_str() << ": "
              << status_scores.description().c_str() << "\n";
#endif
    return;
  }

  auto scores_dims = scores->GetDims();
  const unsigned int num_anchors = scores_dims.at(1); // n = 17640 (640x480)
  const float *scores_ptr = (float *) scores->GetData();
  const float *boxes_ptr = (float *) boxes->GetData();

  bbox_collection.clear();
  unsigned int count = 0;
  for (unsigned int i = 0; i < num_anchors; ++i)
  {
    float confidence = scores_ptr[2 * i + 1];
    if (confidence < score_threshold) continue;
    types::Boxf box;
    box.x1 = boxes_ptr[4 * i + 0] * img_width;
    box.y1 = boxes_ptr[4 * i + 1] * img_height;
    box.x2 = boxes_ptr[4 * i + 2] * img_width;
    box.y2 = boxes_ptr[4 * i + 3] * img_height;
    box.score = confidence;
    box.label_text = "face";
    box.label = 1;
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

void TNNUltraFace::nms(std::vector<types::Boxf> &input, std::vector<types::Boxf> &output,
                       float iou_threshold, unsigned int topk,
                       unsigned int nms_type)
{
  if (nms_type == NMS::BLEND) lite::utils::blending_nms(input, output, iou_threshold, topk);
  else if (nms_type == NMS::OFFSET) lite::utils::offset_nms(input, output, iou_threshold, topk);
  else lite::utils::hard_nms(input, output, iou_threshold, topk);
}







































