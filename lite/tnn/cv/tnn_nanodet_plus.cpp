//
// Created by DefTruth on 2021/12/27.
//

#include "tnn_nanodet_plus.h"
#include "lite/utils.h"

using tnncv::TNNNanoDetPlus;

TNNNanoDetPlus::TNNNanoDetPlus(const std::string &_proto_path,
                               const std::string &_model_path,
                               unsigned int _num_threads) :
    BasicTNNHandler(_proto_path, _model_path, _num_threads)
{
}

void TNNNanoDetPlus::resize_unscale(const cv::Mat &mat, cv::Mat &mat_rs,
                                    int target_height, int target_width,
                                    NanoPlusScaleParams &scale_params)
{
  if (mat.empty()) return;
  int img_height = static_cast<int>(mat.rows);
  int img_width = static_cast<int>(mat.cols);

  mat_rs = cv::Mat(target_height, target_width, CV_8UC3,
                   cv::Scalar(0, 0, 0));
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
  cv::Mat new_unpad_mat;
  // cv::Mat new_unpad_mat = mat.clone(); // may not need clone.
  cv::resize(mat, new_unpad_mat, cv::Size(new_unpad_w, new_unpad_h));
  new_unpad_mat.copyTo(mat_rs(cv::Rect(dw, dh, new_unpad_w, new_unpad_h)));

  // record scale params.
  scale_params.ratio = r;
  scale_params.dw = dw;
  scale_params.dh = dh;
  scale_params.flag = true;
}

void TNNNanoDetPlus::transform(const cv::Mat &mat_rs)
{
  // push into input_mat, BGR
  input_mat = std::make_shared<tnn::Mat>(input_device_type, tnn::N8UC3,
                                         input_shape, (void *) mat_rs.data);
  if (!input_mat->GetData())
  {
#ifdef LITETNN_DEBUG
    std::cout << "input_mat == nullptr! transform failed\n";
#endif
  }
}

void TNNNanoDetPlus::detect(const cv::Mat &mat, std::vector<types::Boxf> &detected_boxes,
                            float score_threshold, float iou_threshold,
                            unsigned int topk, unsigned int nms_type)
{
  if (mat.empty()) return;
  float img_height = static_cast<float>(mat.rows);
  float img_width = static_cast<float>(mat.cols);

  // resize & unscale
  cv::Mat mat_rs;
  NanoPlusScaleParams scale_params;
  this->resize_unscale(mat, mat_rs, input_height, input_width, scale_params);

  if ((!scale_params.flag) || mat_rs.empty()) return;
  // 1. make input mat
  this->transform(mat_rs);
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
  // 4. fetch bounding boxes
  std::vector<types::Boxf> bbox_collection;
  this->generate_bboxes(scale_params, bbox_collection, instance, score_threshold, img_height, img_width);
  // 5. hard|blend|offset nms with topk.
  this->nms(bbox_collection, detected_boxes, iou_threshold, topk, nms_type);
}

void TNNNanoDetPlus::generate_points(unsigned int target_height, unsigned int target_width)
{
  if (center_points_is_update) return;
  // 8, 16, 32, 64
  for (auto stride: strides)
  {
    unsigned int num_grid_w = target_width / stride;
    unsigned int num_grid_h = target_height / stride;

    for (unsigned int g1 = 0; g1 < num_grid_h; ++g1)
    {
      for (unsigned int g0 = 0; g0 < num_grid_w; ++g0)
      {
        float grid0 = (float) g0;
        float grid1 = (float) g1;
#ifdef LITE_WIN32
        NanoPlusCenterPoint point;
        point.grid0 = grid0;
        point.grid1 = grid1;
        point.stride = (float) stride;
        center_points.push_back(point);
#else
        center_points.push_back((NanoPlusCenterPoint) {grid0, grid1, (float) stride});
#endif
      }
    }
  }

  center_points_is_update = true;
}

void TNNNanoDetPlus::generate_bboxes(const NanoPlusScaleParams &scale_params,
                                     std::vector<types::Boxf> &bbox_collection,
                                     std::shared_ptr<tnn::Instance> &_instance,
                                     float score_threshold, float img_height,
                                     float img_width)
{
  std::shared_ptr<tnn::Mat> output_pred;
  tnn::MatConvertParam cvt_param;
  tnn::Status status;

  status = _instance->GetOutputMat(output_pred, cvt_param, "output", output_device_type);   // e.g [1,2125,112]

  if (status != tnn::TNN_OK)
  {
#ifdef LITETNN_DEBUG
    std::cout << "instance->GetOutputMat failed!:"
              << status.description().c_str() << "\n";
#endif
    return;
  }

  this->generate_points(input_height, input_width); // e.g 320 320

  auto output_pred_dims = output_pred->GetDims(); // e.g [1,2125,112]
#ifdef LITETNN_DEBUG
  BasicTNNHandler::print_name_shape("output", output_pred_dims);
#endif
  const unsigned int num_classes = 80;
  const unsigned int num_cls_reg = output_pred_dims.at(2); // 112
  const unsigned int reg_max = (num_cls_reg - num_classes) / 4; // e.g 8=7+1
  const unsigned int num_points = center_points.size();
  const float *output_pred_ptr = (float *) output_pred->GetData();
  float ratio = scale_params.ratio;
  int dw = scale_params.dw;
  int dh = scale_params.dh;

  unsigned int count = 0;

  bbox_collection.clear();
  for (unsigned int i = 0; i < num_points; ++i)
  {
    const float *scores = output_pred_ptr + i * num_cls_reg; // row ptr
    float cls_conf = scores[0];
    unsigned int label = 0;
    for (unsigned int j = 0; j < num_classes; ++j)
    {
      float tmp_conf = scores[j];
      if (tmp_conf > cls_conf)
      {
        cls_conf = tmp_conf;
        label = j;
      }
    } // argmax
    if (cls_conf < score_threshold) continue; // filter

    auto &point = center_points.at(i);
    const float cx = point.grid0; // cx
    const float cy = point.grid1; // cy
    const float s = point.stride; // stride

    const float *logits = output_pred_ptr + i * num_cls_reg + num_classes;  // 32|44...
    std::vector<float> offsets(4);
    for (unsigned int k = 0; k < 4; ++k)
    {
      float offset = 0.f;
      unsigned int max_id;
      auto probs = lite::utils::math::softmax<float>(
          logits + (k * reg_max), reg_max, max_id);
      for (unsigned int l = 0; l < reg_max; ++l)
        offset += (float) l * probs[l];
      offsets[k] = offset;
    }

    float l = offsets[0]; // left
    float t = offsets[1]; // top
    float r = offsets[2]; // right
    float b = offsets[3]; // bottom

    types::Boxf box;
    float x1 = ((cx - l) * s - (float) dw) / ratio;  // cx - l x1
    float y1 = ((cy - t) * s - (float) dh) / ratio;  // cy - t y1
    float x2 = ((cx + r) * s - (float) dw) / ratio;  // cx + r x2
    float y2 = ((cy + b) * s - (float) dh) / ratio;  // cy + b y2
    box.x1 = std::max(0.f, x1);
    box.y1 = std::max(0.f, y1);
    box.x2 = std::min(img_width - 1.f, x2);
    box.y2 = std::min(img_height - 1.f, y2);
    box.score = cls_conf;
    box.label = label;
    box.label_text = class_names[label];
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

void TNNNanoDetPlus::nms(std::vector<types::Boxf> &input, std::vector<types::Boxf> &output,
                         float iou_threshold, unsigned int topk, unsigned int nms_type)
{
  if (nms_type == NMS::BLEND) lite::utils::blending_nms(input, output, iou_threshold, topk);
  else if (nms_type == NMS::OFFSET) lite::utils::offset_nms(input, output, iou_threshold, topk);
  else lite::utils::hard_nms(input, output, iou_threshold, topk);
}

