//
// Created by DefTruth on 2022/3/27.
//

#include "tnn_insectdet.h"
#include "lite/utils.h"

using tnncv::TNNInsectDet;

TNNInsectDet::TNNInsectDet(const std::string &_proto_path,
                           const std::string &_model_path,
                           unsigned int _num_threads) :
    BasicTNNHandler(_proto_path, _model_path, _num_threads)
{
}

void TNNInsectDet::resize_unscale(const cv::Mat &mat, cv::Mat &mat_rs,
                                  int target_height, int target_width,
                                  InsectDetScaleParams &scale_params)
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
  scale_params.ratio = r;
  scale_params.dw = dw;
  scale_params.dh = dh;
  scale_params.flag = true;
}

void TNNInsectDet::transform(const cv::Mat &mat_rs)
{
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

void TNNInsectDet::detect(const cv::Mat &mat, std::vector<types::Boxf> &detected_boxes,
                          float score_threshold, float iou_threshold,
                          unsigned int topk)
{
  if (mat.empty()) return;
  int img_height = static_cast<int>(mat.rows);
  int img_width = static_cast<int>(mat.cols);
  // resize & unscale
  cv::Mat mat_rs;
  InsectDetScaleParams scale_params;
  this->resize_unscale(mat, mat_rs, input_height, input_width, scale_params);

  // 1. make input tensor
  cv::Mat mat_rs_;
  cv::cvtColor(mat_rs, mat_rs_, cv::COLOR_BGR2RGB);
  this->transform(mat_rs_);
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
  // 5. rescale & exclude.
  std::vector<types::Boxf> bbox_collection;
  this->generate_bboxes(scale_params, bbox_collection, instance, score_threshold, img_height, img_width);
  // 6. hard|blend|offset nms with topk.
  this->nms(bbox_collection, detected_boxes, iou_threshold, topk);
}

void TNNInsectDet::generate_bboxes(const InsectDetScaleParams &scale_params,
                                   std::vector<types::Boxf> &bbox_collection,
                                   std::shared_ptr<tnn::Instance> &_instance,
                                   float score_threshold, int img_height,
                                   int img_width)
{
  tnn::MatConvertParam cvt_param;
  std::shared_ptr<tnn::Mat> output;
  tnn::Status status;

  status = _instance->GetOutputMat(output, cvt_param, "output", output_device_type); // [1,N,6]
  if (status != tnn::TNN_OK)
  {
#ifdef LITETNN_DEBUG
    std::cout << "instance->GetOutputMat failed!:" << status.description().c_str() << "\n";
#endif
    return;
  }

  auto output_dims = output->GetDims();
  const unsigned int num_anchors = output_dims.at(1); // n = ?
  const float *output_ptr = (float *) output->GetData();

  float r_ = scale_params.ratio;
  int dw_ = scale_params.dw;
  int dh_ = scale_params.dh;

  bbox_collection.clear();
  unsigned int count = 0;
  for (unsigned int i = 0; i < num_anchors; ++i)
  {
    const float *row_ptr = output_ptr + i * 6;
    float obj_conf = row_ptr[4];
    if (obj_conf < score_threshold) continue; // filter first.
    float cls_conf = row_ptr[5];
    if (cls_conf < score_threshold) continue; // insect score.

    // bounding box
    const float *offsets = row_ptr;
    float cx = offsets[0];
    float cy = offsets[1];
    float w = offsets[2];
    float h = offsets[3];

    types::Boxf box;
    float x1 = ((cx - w / 2.f) - (float) dw_) / r_;
    float y1 = ((cy - h / 2.f) - (float) dh_) / r_;
    float x2 = ((cx + w / 2.f) - (float) dw_) / r_;
    float y2 = ((cy + h / 2.f) - (float) dh_) / r_;
    box.x1 = std::max(0.f, x1);
    box.y1 = std::max(0.f, y1);
    box.x2 = std::min((float) img_width, x2);
    box.y2 = std::min((float) img_height, y2);
    box.score = cls_conf;
    box.label = 1;
    box.label_text = "insect";
    box.flag = true;

    bbox_collection.push_back(box);

    count += 1; // limit boxes for nms.
    if (count > max_nms)
      break;
  }

#if LITETNN_DEBUG
  std::cout << "generate_bboxes num: " << bbox_collection.size() << "\n";
#endif
}

void TNNInsectDet::nms(std::vector<types::Boxf> &input, std::vector<types::Boxf> &output,
                       float iou_threshold, unsigned int topk)
{
  lite::utils::hard_nms(input, output, iou_threshold, topk);
}