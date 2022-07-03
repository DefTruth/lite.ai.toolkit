//
// Created by DefTruth on 2021/11/6.
//

#include "ncnn_yolov5.h"
#include "lite/utils.h"

using ncnncv::NCNNYoloV5;

NCNNYoloV5::NCNNYoloV5(const std::string &_param_path,
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

NCNNYoloV5::~NCNNYoloV5()
{
  if (net) delete net;
  net = nullptr;
}

void NCNNYoloV5::transform(const cv::Mat &mat_rs, ncnn::Mat &in)
{
  // BGR NHWC -> RGB NCHW
  in = ncnn::Mat::from_pixels(mat_rs.data, ncnn::Mat::PIXEL_BGR2RGB, input_width, input_height);
  in.substract_mean_normalize(mean_vals, norm_vals);
}

void NCNNYoloV5::resize_unscale(const cv::Mat &mat, cv::Mat &mat_rs,
                                int target_height, int target_width,
                                YoloV5ScaleParams &scale_params)
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
  cv::Mat new_unpad_mat;
  // cv::Mat new_unpad_mat = mat.clone(); // may not need clone.
  cv::resize(mat, new_unpad_mat, cv::Size(new_unpad_w, new_unpad_h));
  new_unpad_mat.copyTo(mat_rs(cv::Rect(dw, dh, new_unpad_w, new_unpad_h)));

  // record scale params.
  scale_params.r = r;
  scale_params.dw = dw;
  scale_params.dh = dh;
  scale_params.new_unpad_w = new_unpad_w;
  scale_params.new_unpad_h = new_unpad_h;
  scale_params.flag = true;
}

void NCNNYoloV5::detect(const cv::Mat &mat,
                        std::vector<types::Boxf> &detected_boxes,
                        float score_threshold, float iou_threshold,
                        unsigned int topk, unsigned int nms_type)
{
  if (mat.empty()) return;
  int img_height = static_cast<int>(mat.rows);
  int img_width = static_cast<int>(mat.cols);
  // resize & unscale
  cv::Mat mat_rs;
  YoloV5ScaleParams scale_params;
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
  this->generate_bboxes(scale_params, extractor, bbox_collection,
                        score_threshold, img_height, img_width);
  // 5. hard|blend nms with topk.
  this->nms(bbox_collection, detected_boxes, iou_threshold, topk, nms_type);
}

void NCNNYoloV5::generate_anchors(unsigned int target_height, unsigned int target_width)
{
  if (center_anchors_is_update) return;

  for (auto stride : strides)
  {
    unsigned int num_grid_w = target_width / stride;
    unsigned int num_grid_h = target_height / stride;
    std::vector<YoloV5Anchor> anchors;

    if (stride == 8)
    {
      // 0 anchor
      for (unsigned int g1 = 0; g1 < num_grid_h; ++g1)
      {
        for (unsigned int g0 = 0; g0 < num_grid_w; ++g0)
        {
          YoloV5Anchor anchor;
          anchor.grid0 = g0;
          anchor.grid1 = g1;
          anchor.stride = stride;
          anchor.width = 10.f;
          anchor.height = 13.f;
          anchors.push_back(anchor);
        }
      }
      // 1 anchor
      for (unsigned int g1 = 0; g1 < num_grid_h; ++g1)
      {
        for (unsigned int g0 = 0; g0 < num_grid_w; ++g0)
        {
          YoloV5Anchor anchor;
          anchor.grid0 = g0;
          anchor.grid1 = g1;
          anchor.stride = stride;
          anchor.width = 16.f;
          anchor.height = 30.f;
          anchors.push_back(anchor);
        }
      }
      // 2 anchor
      for (unsigned int g1 = 0; g1 < num_grid_h; ++g1)
      {
        for (unsigned int g0 = 0; g0 < num_grid_w; ++g0)
        {
          YoloV5Anchor anchor;
          anchor.grid0 = g0;
          anchor.grid1 = g1;
          anchor.stride = stride;
          anchor.width = 33.f;
          anchor.height = 23.f;
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
          YoloV5Anchor anchor;
          anchor.grid0 = g0;
          anchor.grid1 = g1;
          anchor.stride = stride;
          anchor.width = 30.f;
          anchor.height = 61.f;
          anchors.push_back(anchor);
        }
      }
      // 1 anchor
      for (unsigned int g1 = 0; g1 < num_grid_h; ++g1)
      {
        for (unsigned int g0 = 0; g0 < num_grid_w; ++g0)
        {
          YoloV5Anchor anchor;
          anchor.grid0 = g0;
          anchor.grid1 = g1;
          anchor.stride = stride;
          anchor.width = 62.f;
          anchor.height = 45.f;
          anchors.push_back(anchor);
        }
      }
      // 2 anchor
      for (unsigned int g1 = 0; g1 < num_grid_h; ++g1)
      {
        for (unsigned int g0 = 0; g0 < num_grid_w; ++g0)
        {
          YoloV5Anchor anchor;
          anchor.grid0 = g0;
          anchor.grid1 = g1;
          anchor.stride = stride;
          anchor.width = 59.f;
          anchor.height = 119.f;
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
          YoloV5Anchor anchor;
          anchor.grid0 = g0;
          anchor.grid1 = g1;
          anchor.stride = stride;
          anchor.width = 116.f;
          anchor.height = 90.f;
          anchors.push_back(anchor);
        }
      }
      // 1 anchor
      for (unsigned int g1 = 0; g1 < num_grid_h; ++g1)
      {
        for (unsigned int g0 = 0; g0 < num_grid_w; ++g0)
        {
          YoloV5Anchor anchor;
          anchor.grid0 = g0;
          anchor.grid1 = g1;
          anchor.stride = stride;
          anchor.width = 156.f;
          anchor.height = 198.f;
          anchors.push_back(anchor);
        }
      }
      // 2 anchor
      for (unsigned int g1 = 0; g1 < num_grid_h; ++g1)
      {
        for (unsigned int g0 = 0; g0 < num_grid_w; ++g0)
        {
          YoloV5Anchor anchor;
          anchor.grid0 = g0;
          anchor.grid1 = g1;
          anchor.stride = stride;
          anchor.width = 373.f;
          anchor.height = 326.f;
          anchors.push_back(anchor);
        }
      }
    }
    center_anchors[stride] = anchors;
  }

  center_anchors_is_update = true;
}

void NCNNYoloV5::generate_bboxes(const YoloV5ScaleParams &scale_params,
                                 ncnn::Extractor &extractor,
                                 std::vector<types::Boxf> &bbox_collection,
                                 float score_threshold, float img_height,
                                 float img_width)
{
  // (1,n,85=5+80=cxcy+cwch+obj_conf+cls_conf)
  ncnn::Mat det_stride_8, det_stride_16, det_stride_32;
  extractor.extract("det_stride_8", det_stride_8);
  extractor.extract("det_stride_16", det_stride_16);
  extractor.extract("det_stride_32", det_stride_32);

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
}

// inner function
static inline float sigmoid(float x)
{
  return static_cast<float>(1.f / (1.f + std::exp(-x)));
}

void NCNNYoloV5::generate_bboxes_single_stride(const YoloV5ScaleParams &scale_params,
                                               ncnn::Mat &det_pred,
                                               unsigned int stride,
                                               float score_threshold,
                                               float img_height, float img_width,
                                               std::vector<types::Boxf> &bbox_collection)
{
  unsigned int nms_pre_ = (stride / 8) * nms_pre; // 1 * 1000,2*1000,...
  nms_pre_ = nms_pre_ >= nms_pre ? nms_pre_ : nms_pre;

  const unsigned int f_h = (unsigned int) input_height / stride;
  const unsigned int f_w = (unsigned int) input_width / stride;
  // e.g, 3*80*80 + 3*40*40 + 3*20*20 = 25200
  const unsigned int num_anchors = 3 * f_h * f_w;
  const unsigned int num_classes = 80;

  float r_ = scale_params.r;
  int dw_ = scale_params.dw;
  int dh_ = scale_params.dh;

  // have c=3 indicate 3 anchors at one grid
  unsigned int count = 0;
  auto &stride_anchors = center_anchors[stride];

  for (unsigned int i = 0; i < num_anchors; ++i)
  {
    const float *offset_obj_cls_ptr = (float *) det_pred.data + (i * (num_classes + 5));
    float obj_conf = sigmoid(offset_obj_cls_ptr[4]);
    if (obj_conf < score_threshold) continue; // filter first.

    float cls_conf = sigmoid(offset_obj_cls_ptr[5]);
    unsigned int label = 0;  // 80 class
    for (unsigned int j = 0; j < num_classes; ++j)
    {
      float tmp_conf = sigmoid(offset_obj_cls_ptr[j + 5]);
      if (tmp_conf > cls_conf)
      {
        cls_conf = tmp_conf;
        label = j;
      }
    } // argmax

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

    float cx = (dx * 2.f - 0.5f + (float) grid0) * (float) stride;
    float cy = (dy * 2.f - 0.5f + (float) grid1) * (float) stride;
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
    box.x2 = std::min(x2, (float) img_width - 1.f);
    box.y2 = std::min(y2, (float) img_height - 1.f);
    box.score = conf;
    box.label = label;
    box.label_text = class_names[label];
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

void NCNNYoloV5::nms(std::vector<types::Boxf> &input, std::vector<types::Boxf> &output,
                     float iou_threshold, unsigned int topk,
                     unsigned int nms_type)
{
  if (nms_type == NMS::BLEND) lite::utils::blending_nms(input, output, iou_threshold, topk);
  else if (nms_type == NMS::OFFSET) lite::utils::offset_nms(input, output, iou_threshold, topk);
  else lite::utils::hard_nms(input, output, iou_threshold, topk);
}

void NCNNYoloV5::print_debug_string()
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