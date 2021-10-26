//
// Created by DefTruth on 2021/9/14.
//
#include "yolop.h"
#include "lite/ort/core/ort_utils.h"
#include "lite/utils.h"

using ortcv::YOLOP;

void YOLOP::resize_unscale(const cv::Mat &mat, cv::Mat &mat_rs,
                           int target_height, int target_width,
                           YOLOPScaleParams &scale_params)
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

Ort::Value YOLOP::transform(const cv::Mat &mat_rs)
{
  cv::Mat canva = mat_rs.clone();
  cv::cvtColor(canva, canva, cv::COLOR_BGR2RGB);

  // (1,3,640,640) 1xCXHXW
  ortcv::utils::transform::normalize_inplace(canva, mean_vals, scale_vals); // float32
  return ortcv::utils::transform::create_tensor(
      canva, input_node_dims, memory_info_handler,
      input_values_handler, ortcv::utils::transform::CHW);
}

void YOLOP::detect(const cv::Mat &mat,
                   std::vector<types::Boxf> &detected_boxes,
                   types::SegmentContent &da_seg_content,
                   types::SegmentContent &ll_seg_content,
                   float score_threshold, float iou_threshold,
                   unsigned int topk, unsigned int nms_type)
{
  if (mat.empty()) return;
  float img_height = static_cast<float>(mat.rows);
  float img_width = static_cast<float>(mat.cols);
  const int target_height = input_node_dims.at(2);
  const int target_width = input_node_dims.at(3);

  // resize & unscale
  cv::Mat mat_rs;
  YOLOPScaleParams scale_params;
  this->resize_unscale(mat, mat_rs, target_height, target_width, scale_params);

  if ((!scale_params.flag) || mat_rs.empty()) return;

  // 1. make input tensor
  Ort::Value input_tensor = this->transform(mat_rs);
  // 2. inference scores & boxes.
  auto output_tensors = ort_session->Run(
      Ort::RunOptions{nullptr}, input_node_names.data(),
      &input_tensor, 1, output_node_names.data(), num_outputs
  );  // det_out, drive_area_seg, lane_line_seg

  // 3. rescale & fetch da|ll seg.
  std::vector<types::Boxf> bbox_collection;
  this->generate_bboxes_da_ll(scale_params, output_tensors, bbox_collection,
                              da_seg_content, ll_seg_content, score_threshold,
                              img_height, img_width);

  // 4. hard|blend nms with topk.
  this->nms(bbox_collection, detected_boxes, iou_threshold, topk, nms_type);
}

void YOLOP::generate_bboxes_da_ll(const YOLOPScaleParams &scale_params,
                                  std::vector<Ort::Value> &output_tensors,
                                  std::vector<types::Boxf> &bbox_collection,
                                  types::SegmentContent &da_seg_content,
                                  types::SegmentContent &ll_seg_content,
                                  float score_threshold, float img_height,
                                  float img_width)
{
  Ort::Value &det_out = output_tensors.at(0); // (1,n,6=5+1=cxcy+cwch+obj_conf+cls_conf)
  Ort::Value &da_seg_out = output_tensors.at(1); // (1,2,640,640)
  Ort::Value &ll_seg_out = output_tensors.at(2); // (1,2,640,640)
  auto det_dims = output_node_dims.at(0); // (1,n,6)
  const unsigned int num_anchors = det_dims.at(1); // n = ?

  float r = scale_params.r;
  int dw = scale_params.dw;
  int dh = scale_params.dh;
  int new_unpad_w = scale_params.new_unpad_w;
  int new_unpad_h = scale_params.new_unpad_h;

  // generate bounding boxes.
  bbox_collection.clear();
  unsigned int count = 0;
  for (unsigned int i = 0; i < num_anchors; ++i)
  {
    float obj_conf = det_out.At<float>({0, i, 4});
    if (obj_conf < score_threshold) continue; // filter first.

    unsigned int label = 1;  // 1 class only
    float cls_conf = det_out.At<float>({0, i, 5});
    float conf = obj_conf * cls_conf; // cls_conf (0.,1.)
    if (conf < score_threshold) continue; // filter

    float cx = det_out.At<float>({0, i, 0});
    float cy = det_out.At<float>({0, i, 1});
    float w = det_out.At<float>({0, i, 2});
    float h = det_out.At<float>({0, i, 3});

    types::Boxf box;
    // de-padding & rescaling
    box.x1 = ((cx - w / 2.f) - (float) dw) / r;
    box.y1 = ((cy - h / 2.f) - (float) dh) / r;
    box.x2 = ((cx + w / 2.f) - (float) dw) / r;
    box.y2 = ((cy + h / 2.f) - (float) dh) / r;
    box.score = conf;
    box.label = label;
    box.label_text = "traffic car";
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

  // generate da && ll seg.
  da_seg_content.names_map.clear();
  da_seg_content.class_mat = cv::Mat(new_unpad_h, new_unpad_w, CV_8UC1, cv::Scalar(0));
  da_seg_content.color_mat = cv::Mat(new_unpad_h, new_unpad_w, CV_8UC3, cv::Scalar(0, 0, 0));
  ll_seg_content.names_map.clear();
  ll_seg_content.class_mat = cv::Mat(new_unpad_h, new_unpad_w, CV_8UC1, cv::Scalar(0));
  ll_seg_content.color_mat = cv::Mat(new_unpad_h, new_unpad_w, CV_8UC3, cv::Scalar(0, 0, 0));

  for (int i = dh; i < dh + new_unpad_h; ++i)
  {
    // row ptr.
    uchar *da_p_class = da_seg_content.class_mat.ptr<uchar>(i - dh);
    uchar *ll_p_class = ll_seg_content.class_mat.ptr<uchar>(i - dh);
    cv::Vec3b *da_p_color = da_seg_content.color_mat.ptr<cv::Vec3b>(i - dh);
    cv::Vec3b *ll_p_color = ll_seg_content.color_mat.ptr<cv::Vec3b>(i - dh);

    for (int j = dw; j < dw + new_unpad_w; ++j)
    {
      // argmax
      float da_bg_prob = da_seg_out.At<float>({0, 0, i, j});
      float da_fg_prob = da_seg_out.At<float>({0, 1, i, j});
      float ll_bg_prob = ll_seg_out.At<float>({0, 0, i, j});
      float ll_fg_prob = ll_seg_out.At<float>({0, 1, i, j});
      unsigned int da_label = da_bg_prob < da_fg_prob ? 1 : 0;
      unsigned int ll_label = ll_bg_prob < ll_fg_prob ? 1 : 0;

      if (da_label == 1)
      {
        // assign label for pixel(i,j)
        da_p_class[j - dw] = 1 * 255;  // 255 indicate drivable area, for post resize
        // assign color for detected class at pixel(i,j).
        da_p_color[j - dw][0] = 0;
        da_p_color[j - dw][1] = 255;  // green
        da_p_color[j - dw][2] = 0;
        // assign names map
        da_seg_content.names_map[255] = "drivable area";
      }

      if (ll_label == 1)
      {
        // assign label for pixel(i,j)
        ll_p_class[j - dw] = 1 * 255;  // 255 indicate lane line, for post resize
        // assign color for detected class at pixel(i,j).
        ll_p_color[j - dw][0] = 0;
        ll_p_color[j - dw][1] = 0;
        ll_p_color[j - dw][2] = 255;  // red
        // assign names map
        ll_seg_content.names_map[255] = "lane line";
      }

    }
  }
  // resize to original size.
  const unsigned int img_h = static_cast<unsigned int>(img_height);
  const unsigned int img_w = static_cast<unsigned int>(img_width);
  // da_seg_mask 255 or 0
  cv::resize(da_seg_content.class_mat, da_seg_content.class_mat,
             cv::Size(img_w, img_h), cv::INTER_LINEAR);
  cv::resize(da_seg_content.color_mat, da_seg_content.color_mat,
             cv::Size(img_w, img_h), cv::INTER_LINEAR);
  // ll_seg_mask 255 or 0
  cv::resize(ll_seg_content.class_mat, ll_seg_content.class_mat,
             cv::Size(img_w, img_h), cv::INTER_LINEAR);
  cv::resize(ll_seg_content.color_mat, ll_seg_content.color_mat,
             cv::Size(img_w, img_h), cv::INTER_LINEAR);

  da_seg_content.flag = true;
  ll_seg_content.flag = true;

}

void YOLOP::nms(std::vector<types::Boxf> &input, std::vector<types::Boxf> &output,
                float iou_threshold, unsigned int topk, unsigned int nms_type)
{
  if (nms_type == NMS::BLEND) lite::utils::blending_nms(input, output, iou_threshold, topk);
  else if (nms_type == NMS::OFFSET) lite::utils::offset_nms(input, output, iou_threshold, topk);
  else lite::utils::hard_nms(input, output, iou_threshold, topk);
}