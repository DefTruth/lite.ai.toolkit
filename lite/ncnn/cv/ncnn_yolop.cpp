//
// Created by DefTruth on 2021/10/18.
//

#include "ncnn_yolop.h"
#include "lite/utils.h"

using ncnncv::NCNNYOLOP;

NCNNYOLOP::NCNNYOLOP(const std::string &_param_path,
                     const std::string &_bin_path,
                     unsigned int _num_threads,
                     int _input_height,
                     int _input_width) :
    log_id(_param_path.data()), param_path(_param_path.data()),
    bin_path(_bin_path.data()), num_threads(_num_threads),
    input_height(_input_height), input_width(_input_width)
{
  net = new ncnn::Net();
  // init net, change this setting for better performance.
  net->opt.use_fp16_arithmetic = false;
  net->opt.use_vulkan_compute = false; // default
  // setup Focus in yolov5
  net->register_custom_layer("YoloV5Focus", YoloV5Focus_layer_creator);
  net->load_param(param_path);
  net->load_model(bin_path);
#ifdef LITENCNN_DEBUG
  this->print_debug_string();
#endif
}

NCNNYOLOP::~NCNNYOLOP()
{
  if (net) delete net;
  net = nullptr;
}

void NCNNYOLOP::transform(const cv::Mat &mat_rs, ncnn::Mat &in)
{
  // BGR NHWC -> RGB NCHW
  in = ncnn::Mat::from_pixels(mat_rs.data, ncnn::Mat::PIXEL_BGR2RGB, input_width, input_height);
  in.substract_mean_normalize(mean_vals, norm_vals);
}

void NCNNYOLOP::resize_unscale(const cv::Mat &mat, cv::Mat &mat_rs,
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

void NCNNYOLOP::detect(const cv::Mat &mat,
                       std::vector<types::Boxf> &detected_boxes,
                       types::SegmentContent &da_seg_content,
                       types::SegmentContent &ll_seg_content,
                       float score_threshold, float iou_threshold,
                       unsigned int topk, unsigned int nms_type)
{
  if (mat.empty()) return;
  int img_height = static_cast<int>(mat.rows);
  int img_width = static_cast<int>(mat.cols);
  // resize & unscale
  cv::Mat mat_rs;
  YOLOPScaleParams scale_params;
  this->resize_unscale(mat, mat_rs, input_height, input_width, scale_params);
  // 1. make input tensor
  ncnn::Mat input;
  this->transform(mat_rs, input);
  // 2. inference & extract
  auto extractor = net->create_extractor();
  extractor.set_light_mode(false);  // default
  extractor.set_num_threads(num_threads);
  extractor.input("images", input);
  // 4. rescale & fetch da|ll seg.
  std::vector<types::Boxf> bbox_collection;
  this->generate_bboxes_da_ll(scale_params, extractor, bbox_collection,
                              da_seg_content, ll_seg_content, score_threshold,
                              img_height, img_width);
  // 5. hard|blend nms with topk.
  this->nms(bbox_collection, detected_boxes, iou_threshold, topk, nms_type);
}

void NCNNYOLOP::generate_anchors(unsigned int target_height, unsigned int target_width)
{
  if (center_anchors_is_update) return;

  for (auto stride : strides)
  {
    unsigned int num_grid_w = target_width / stride;
    unsigned int num_grid_h = target_height / stride;
    std::vector<YOLOPAnchor> anchors;

    if (stride == 8)
    {
      // 0 anchor
      for (unsigned int g1 = 0; g1 < num_grid_h; ++g1)
      {
        for (unsigned int g0 = 0; g0 < num_grid_w; ++g0)
        {
          YOLOPAnchor anchor;
          anchor.grid0 = g0;
          anchor.grid1 = g1;
          anchor.stride = stride;
          anchor.width = 3.f;
          anchor.height = 9.f;
          anchors.push_back(anchor);
        }
      }
      // 1 anchor
      for (unsigned int g1 = 0; g1 < num_grid_h; ++g1)
      {
        for (unsigned int g0 = 0; g0 < num_grid_w; ++g0)
        {
          YOLOPAnchor anchor;
          anchor.grid0 = g0;
          anchor.grid1 = g1;
          anchor.stride = stride;
          anchor.width = 5.f;
          anchor.height = 11.f;
          anchors.push_back(anchor);
        }
      }
      // 2 anchor
      for (unsigned int g1 = 0; g1 < num_grid_h; ++g1)
      {
        for (unsigned int g0 = 0; g0 < num_grid_w; ++g0)
        {
          YOLOPAnchor anchor;
          anchor.grid0 = g0;
          anchor.grid1 = g1;
          anchor.stride = stride;
          anchor.width = 4.f;
          anchor.height = 20.f;
          anchors.push_back(anchor);
        }
      }
    } // 16
    else if (stride == 16)
    {
      // 0 anchor
      for (unsigned int g1 = 0; g1 < num_grid_h; ++g1)
      {
        for (unsigned int g0 = 0; g0 < num_grid_w; ++g0)
        {
          YOLOPAnchor anchor;
          anchor.grid0 = g0;
          anchor.grid1 = g1;
          anchor.stride = stride;
          anchor.width = 7.f;
          anchor.height = 18.f;
          anchors.push_back(anchor);
        }
      }
      // 1 anchor
      for (unsigned int g1 = 0; g1 < num_grid_h; ++g1)
      {
        for (unsigned int g0 = 0; g0 < num_grid_w; ++g0)
        {
          YOLOPAnchor anchor;
          anchor.grid0 = g0;
          anchor.grid1 = g1;
          anchor.stride = stride;
          anchor.width = 6.f;
          anchor.height = 39.f;
          anchors.push_back(anchor);
        }
      }
      // 2 anchor
      for (unsigned int g1 = 0; g1 < num_grid_h; ++g1)
      {
        for (unsigned int g0 = 0; g0 < num_grid_w; ++g0)
        {
          YOLOPAnchor anchor;
          anchor.grid0 = g0;
          anchor.grid1 = g1;
          anchor.stride = stride;
          anchor.width = 12.f;
          anchor.height = 31.f;
          anchors.push_back(anchor);
        }
      }
    } // 32
    else
    {
      // 0 anchor
      for (unsigned int g1 = 0; g1 < num_grid_h; ++g1)
      {
        for (unsigned int g0 = 0; g0 < num_grid_w; ++g0)
        {
          YOLOPAnchor anchor;
          anchor.grid0 = g0;
          anchor.grid1 = g1;
          anchor.stride = stride;
          anchor.width = 19.f;
          anchor.height = 50.f;
          anchors.push_back(anchor);
        }
      }
      // 1 anchor
      for (unsigned int g1 = 0; g1 < num_grid_h; ++g1)
      {
        for (unsigned int g0 = 0; g0 < num_grid_w; ++g0)
        {
          YOLOPAnchor anchor;
          anchor.grid0 = g0;
          anchor.grid1 = g1;
          anchor.stride = stride;
          anchor.width = 38.f;
          anchor.height = 81.f;
          anchors.push_back(anchor);
        }
      }
      // 2 anchor
      for (unsigned int g1 = 0; g1 < num_grid_h; ++g1)
      {
        for (unsigned int g0 = 0; g0 < num_grid_w; ++g0)
        {
          YOLOPAnchor anchor;
          anchor.grid0 = g0;
          anchor.grid1 = g1;
          anchor.stride = stride;
          anchor.width = 68.f;
          anchor.height = 157.f;
          anchors.push_back(anchor);
        }
      }
    }
    center_anchors[stride] = anchors;
  }

  center_anchors_is_update = true;
}

void NCNNYOLOP::generate_bboxes_da_ll(const YOLOPScaleParams &scale_params,
                                      ncnn::Extractor &extractor,
                                      std::vector<types::Boxf> &bbox_collection,
                                      types::SegmentContent &da_seg_content,
                                      types::SegmentContent &ll_seg_content,
                                      float score_threshold, float img_height,
                                      float img_width)
{
  // (1,n,6=5+1=cxcy+cwch+obj_conf+cls_conf) (1,2,640,640) (1,2,640,640)
  ncnn::Mat det_stride_8, det_stride_16, det_stride_32, da_seg_out, ll_seg_out;
  extractor.extract("det_stride_8", det_stride_8);
  extractor.extract("det_stride_16", det_stride_16);
  extractor.extract("det_stride_32", det_stride_32);
  extractor.extract("drive_area_seg", da_seg_out);
  extractor.extract("lane_line_seg", ll_seg_out);

  this->generate_anchors(input_height, input_width);

  // generate bounding boxes.
  bbox_collection.clear();
  this->generate_bboxes_single_stride(scale_params, det_stride_8, 8, score_threshold,
                                      img_height, img_width, bbox_collection);
  this->generate_bboxes_single_stride(scale_params, det_stride_16, 16, score_threshold,
                                      img_height, img_width, bbox_collection);
  this->generate_bboxes_single_stride(scale_params, det_stride_32, 32, score_threshold,
                                      img_height, img_width, bbox_collection);
#if LITENCNN_DEBUG
  std::cout << "generate_bboxes num: " << bbox_collection.size() << "\n";
#endif

  int dw = scale_params.dw;
  int dh = scale_params.dh;
  int new_unpad_w = scale_params.new_unpad_w;
  int new_unpad_h = scale_params.new_unpad_h;
  // generate da && ll seg.
  da_seg_content.names_map.clear();
  da_seg_content.class_mat = cv::Mat(new_unpad_h, new_unpad_w, CV_8UC1, cv::Scalar(0));
  da_seg_content.color_mat = cv::Mat(new_unpad_h, new_unpad_w, CV_8UC3, cv::Scalar(0, 0, 0));
  ll_seg_content.names_map.clear();
  ll_seg_content.class_mat = cv::Mat(new_unpad_h, new_unpad_w, CV_8UC1, cv::Scalar(0));
  ll_seg_content.color_mat = cv::Mat(new_unpad_h, new_unpad_w, CV_8UC3, cv::Scalar(0, 0, 0));

  const unsigned int channel_step = input_height * input_width;
  const float *da_seg_bg_ptr = (float *) da_seg_out.data; // background
  const float *da_seg_fg_ptr = (float *) da_seg_out.data + channel_step; // foreground
  const float *ll_seg_bg_ptr = (float *) ll_seg_out.data; // background
  const float *ll_seg_fg_ptr = (float *) ll_seg_out.data + channel_step; // foreground

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
      float da_bg_prob = da_seg_bg_ptr[i * input_height + j];
      float da_fg_prob = da_seg_fg_ptr[i * input_height + j];
      float ll_bg_prob = ll_seg_bg_ptr[i * input_height + j];
      float ll_fg_prob = ll_seg_fg_ptr[i * input_height + j];
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

// inner function
static inline float sigmoid(float x)
{
  return static_cast<float>(1.f / (1.f + std::exp(-x)));
}

// reference: https://github.com/Tencent/ncnn/blob/master/examples/yolov5.cpp
void NCNNYOLOP::generate_bboxes_single_stride(const YOLOPScaleParams &scale_params,
                                              ncnn::Mat &det_pred,
                                              unsigned int stride,
                                              float score_threshold,
                                              float img_height,
                                              float img_width,
                                              std::vector<types::Boxf> &bbox_collection)
{
  unsigned int nms_pre_ = (stride / 8) * nms_pre; // 1 * 1000,2*1000,...
  nms_pre_ = nms_pre_ >= nms_pre ? nms_pre_ : nms_pre;

  const unsigned int f_h = (unsigned int) input_height / stride;
  const unsigned int f_w = (unsigned int) input_width / stride;
  // e.g, 3*80*80 + 3*40*40 + 3*20*20 = 25200
  const unsigned int num_anchors = 3 * f_h * f_w;

  float r_ = scale_params.r;
  int dw_ = scale_params.dw;
  int dh_ = scale_params.dh;

  // have c=3 indicate 3 anchors at one grid
  unsigned int count = 0;
  auto &stride_anchors = center_anchors[stride];

  for (unsigned int i = 0; i < num_anchors; ++i)
  {
    const float *offset_obj_cls_ptr = (float *) det_pred.data + (i * 6);
    float obj_conf = sigmoid(offset_obj_cls_ptr[4]);
    if (obj_conf < score_threshold) continue; // filter first.

    unsigned int label = 1;  // 1 class only
    float cls_conf = sigmoid(offset_obj_cls_ptr[5]);
    float conf = obj_conf * cls_conf; // cls_conf (0.,1.)
    if (conf < score_threshold) continue; // filter

    int grid0 = stride_anchors.at(i).grid0; // w
    int grid1 = stride_anchors.at(i).grid1; // h
    float anchor_w = stride_anchors.at(i).width;
    float anchor_h = stride_anchors.at(i).height;

    float dx = sigmoid(offset_obj_cls_ptr[0]);
    float dy = sigmoid(offset_obj_cls_ptr[1]);
    float dw = sigmoid(offset_obj_cls_ptr[2]);
    float dh = sigmoid(offset_obj_cls_ptr[3]);

    float cx = (dx * 2.f - 0.5f + (float)grid0) * (float)stride;
    float cy = (dy * 2.f - 0.5f + (float)grid1) * (float)stride;
    float w = std::pow(dw * 2.f, 2) * anchor_w;
    float h = std::pow(dh * 2.f, 2) * anchor_h;

    float x1 = ((cx - w / 2.f) - (float) dw_) / r_;
    float y1 = ((cy - h / 2.f) - (float) dh_) / r_;
    float x2 = ((cx + w / 2.f) - (float) dw_) / r_;
    float y2 = ((cy + h / 2.f) - (float) dh_) / r_;

    types::Boxf box;
    // de-padding & rescaling
    box.x1 = std::max(0.f, x1);
    box.y1 = std::max(0.f, y1);
    box.x2 = std::min(x2, (float) img_width);
    box.y2 = std::min(y2, (float) img_height);
    box.score = conf;
    box.label = label;
    box.label_text = "traffic car";
    box.flag = true;
    bbox_collection.push_back(box);

    count += 1; // limit boxes for nms.
    if (count > max_nms)
      break;
  }

  if (bbox_collection.size() > nms_pre_)
  {
    std::sort(bbox_collection.begin(), bbox_collection.end(),
              [](const types::Boxf &a, const types::Boxf &b)
              { return a.score > b.score; }); // sort inplace
    // trunc
    bbox_collection.resize(nms_pre_);
  }

}

void NCNNYOLOP::nms(std::vector<types::Boxf> &input, std::vector<types::Boxf> &output,
                    float iou_threshold, unsigned int topk,
                    unsigned int nms_type)
{
  if (nms_type == NMS::BLEND) lite::utils::blending_nms(input, output, iou_threshold, topk);
  else if (nms_type == NMS::OFFSET) lite::utils::offset_nms(input, output, iou_threshold, topk);
  else lite::utils::hard_nms(input, output, iou_threshold, topk);
}

void NCNNYOLOP::print_debug_string()
{
  std::cout << "LITENCNN_DEBUG LogId: " << log_id << "\n";
  input_indexes = net->input_indexes();
  output_indexes = net->output_indexes();
#ifdef NCNN_STRING
  input_names = net->input_names();
  output_names = net->output_names();
#endif
  std::cout << "=============== Input-Dims ==============\n";
  for (int i = 0; i < input_indexes.size(); ++i)
  {
    std::cout << "Input: ";
    auto tmp_in_blob = net->blobs().at(input_indexes.at(i));
#ifdef NCNN_STRING
    std::cout << input_names.at(i) << ": ";
#endif
    std::cout << "shape: c=" << tmp_in_blob.shape.c
              << " h=" << tmp_in_blob.shape.h << " w=" << tmp_in_blob.shape.w << "\n";
  }

  std::cout << "=============== Output-Dims ==============\n";
  for (int i = 0; i < output_indexes.size(); ++i)
  {
    auto tmp_out_blob = net->blobs().at(output_indexes.at(i));
    std::cout << "Output: ";
#ifdef NCNN_STRING
    std::cout << output_names.at(i) << ": ";
#endif
    std::cout << "shape: c=" << tmp_out_blob.shape.c
              << " h=" << tmp_out_blob.shape.h << " w=" << tmp_out_blob.shape.w << "\n";
  }
  std::cout << "========================================\n";
}