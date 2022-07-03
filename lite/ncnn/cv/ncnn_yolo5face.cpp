//
// Created by DefTruth on 2022/1/16.
//

#include "ncnn_yolo5face.h"

using ncnncv::NCNNYOLO5Face;

NCNNYOLO5Face::NCNNYOLO5Face(const std::string &_param_path,
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
  // yolo5face --> no Focus layer in yolo5face
  net->load_param(param_path);
  net->load_model(bin_path);
#ifdef LITENCNN_DEBUG
  this->print_debug_string();
#endif
}

NCNNYOLO5Face::~NCNNYOLO5Face()
{
  if (net) delete net;
  net = nullptr;
}

void NCNNYOLO5Face::transform(const cv::Mat &mat_rs, ncnn::Mat &in)
{
  // BGR NHWC -> RGB NCHW
  in = ncnn::Mat::from_pixels(mat_rs.data, ncnn::Mat::PIXEL_BGR2RGB, input_width, input_height);
  in.substract_mean_normalize(mean_vals, norm_vals);
}

void NCNNYOLO5Face::resize_unscale(const cv::Mat &mat, cv::Mat &mat_rs,
                                   int target_height, int target_width,
                                   YOLO5FaceScaleParams &scale_params)
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

void NCNNYOLO5Face::detect(const cv::Mat &mat, std::vector<types::BoxfWithLandmarks> &detected_boxes_kps,
                           float score_threshold, float iou_threshold, unsigned int topk)
{
  if (mat.empty()) return;
  int img_height = static_cast<int>(mat.rows);
  int img_width = static_cast<int>(mat.cols);
  // resize & unscale
  cv::Mat mat_rs;
  YOLO5FaceScaleParams scale_params;
  this->resize_unscale(mat, mat_rs, input_height, input_width, scale_params);
  // 1. make input tensor
  ncnn::Mat input;
  this->transform(mat_rs, input);
  // 2. inference & extract
  auto extractor = net->create_extractor();
  extractor.set_light_mode(false);  // default
  extractor.set_num_threads(num_threads);
  extractor.input("input", input);

  // 3. rescale & exclude.
  std::vector<types::BoxfWithLandmarks> bbox_kps_collection;
  this->generate_bboxes_kps(scale_params, bbox_kps_collection, extractor,
                            score_threshold, img_height, img_width);
  // 4. hard nms with topk.
  this->nms_bboxes_kps(bbox_kps_collection, detected_boxes_kps, iou_threshold, topk);
}

void NCNNYOLO5Face::generate_anchors(unsigned int target_height, unsigned int target_width)
{
  if (center_anchors_is_update) return;

  for (auto stride : strides)
  {
    unsigned int num_grid_w = target_width / stride;
    unsigned int num_grid_h = target_height / stride;
    std::vector<YOLO5FaceAnchor> anchors;

    if (stride == 8)
    {
      // 0 anchor
      for (unsigned int g1 = 0; g1 < num_grid_h; ++g1)
      {
        for (unsigned int g0 = 0; g0 < num_grid_w; ++g0)
        {
          YOLO5FaceAnchor anchor;
          anchor.grid0 = g0;
          anchor.grid1 = g1;
          anchor.stride = stride;
          anchor.width = 4.f;
          anchor.height = 5.f;
          anchors.push_back(anchor);
        }
      }
      // 1 anchor
      for (unsigned int g1 = 0; g1 < num_grid_h; ++g1)
      {
        for (unsigned int g0 = 0; g0 < num_grid_w; ++g0)
        {
          YOLO5FaceAnchor anchor;
          anchor.grid0 = g0;
          anchor.grid1 = g1;
          anchor.stride = stride;
          anchor.width = 8.f;
          anchor.height = 10.f;
          anchors.push_back(anchor);
        }
      }
      // 2 anchor
      for (unsigned int g1 = 0; g1 < num_grid_h; ++g1)
      {
        for (unsigned int g0 = 0; g0 < num_grid_w; ++g0)
        {
          YOLO5FaceAnchor anchor;
          anchor.grid0 = g0;
          anchor.grid1 = g1;
          anchor.stride = stride;
          anchor.width = 13.f;
          anchor.height = 16.f;
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
          YOLO5FaceAnchor anchor;
          anchor.grid0 = g0;
          anchor.grid1 = g1;
          anchor.stride = stride;
          anchor.width = 23.f;
          anchor.height = 29.f;
          anchors.push_back(anchor);
        }
      }
      // 1 anchor
      for (unsigned int g1 = 0; g1 < num_grid_h; ++g1)
      {
        for (unsigned int g0 = 0; g0 < num_grid_w; ++g0)
        {
          YOLO5FaceAnchor anchor;
          anchor.grid0 = g0;
          anchor.grid1 = g1;
          anchor.stride = stride;
          anchor.width = 43.f;
          anchor.height = 55.f;
          anchors.push_back(anchor);
        }
      }
      // 2 anchor
      for (unsigned int g1 = 0; g1 < num_grid_h; ++g1)
      {
        for (unsigned int g0 = 0; g0 < num_grid_w; ++g0)
        {
          YOLO5FaceAnchor anchor;
          anchor.grid0 = g0;
          anchor.grid1 = g1;
          anchor.stride = stride;
          anchor.width = 73.f;
          anchor.height = 105.f;
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
          YOLO5FaceAnchor anchor;
          anchor.grid0 = g0;
          anchor.grid1 = g1;
          anchor.stride = stride;
          anchor.width = 146.f;
          anchor.height = 217.f;
          anchors.push_back(anchor);
        }
      }
      // 1 anchor
      for (unsigned int g1 = 0; g1 < num_grid_h; ++g1)
      {
        for (unsigned int g0 = 0; g0 < num_grid_w; ++g0)
        {
          YOLO5FaceAnchor anchor;
          anchor.grid0 = g0;
          anchor.grid1 = g1;
          anchor.stride = stride;
          anchor.width = 231.f;
          anchor.height = 300.f;
          anchors.push_back(anchor);
        }
      }
      // 2 anchor
      for (unsigned int g1 = 0; g1 < num_grid_h; ++g1)
      {
        for (unsigned int g0 = 0; g0 < num_grid_w; ++g0)
        {
          YOLO5FaceAnchor anchor;
          anchor.grid0 = g0;
          anchor.grid1 = g1;
          anchor.stride = stride;
          anchor.width = 335.f;
          anchor.height = 433.f;
          anchors.push_back(anchor);
        }
      }
    }
    center_anchors[stride] = anchors;
  }

  center_anchors_is_update = true;
}

void NCNNYOLO5Face::generate_bboxes_kps(const YOLO5FaceScaleParams &scale_params,
                                        std::vector<types::BoxfWithLandmarks> &bbox_kps_collection,
                                        ncnn::Extractor &extractor, float score_threshold,
                                        float img_height, float img_width)
{
// (1,n,16=4+1+10+1=cxcy+cwch+obj_conf+5kps+cls_conf)
  ncnn::Mat det_stride_8, det_stride_16, det_stride_32;
  extractor.extract("det_stride_8", det_stride_8);
  extractor.extract("det_stride_16", det_stride_16);
  extractor.extract("det_stride_32", det_stride_32);

  this->generate_anchors(input_height, input_width);

  // generate bounding boxes.
  bbox_kps_collection.clear();

  this->generate_bboxes_kps_single_stride(scale_params, det_stride_8, 8, score_threshold,
                                          img_height, img_width, bbox_kps_collection);
  this->generate_bboxes_kps_single_stride(scale_params, det_stride_16, 16, score_threshold,
                                          img_height, img_width, bbox_kps_collection);
  this->generate_bboxes_kps_single_stride(scale_params, det_stride_32, 32, score_threshold,
                                          img_height, img_width, bbox_kps_collection);
#if LITENCNN_DEBUG
  std::cout << "generate_bboxes_kps num: " << bbox_kps_collection.size() << "\n";
#endif
}

// inner function
static inline float sigmoid(float x)
{
  return static_cast<float>(1.f / (1.f + std::exp(-x)));
}

void NCNNYOLO5Face::generate_bboxes_kps_single_stride(
    const YOLO5FaceScaleParams &scale_params,
    ncnn::Mat &det_pred, unsigned int stride,
    float score_threshold, float img_height, float img_width,
    std::vector<types::BoxfWithLandmarks> &bbox_kps_collection)
{
  unsigned int nms_pre_ = (stride / 8) * nms_pre; // 1 * 1000,2*1000,...
  nms_pre_ = nms_pre_ >= nms_pre ? nms_pre_ : nms_pre;

  const unsigned int f_h = (unsigned int) input_height / stride;
  const unsigned int f_w = (unsigned int) input_width / stride;
  // e.g, 3*80*80 + 3*40*40 + 3*20*20 = 25200
  const unsigned int num_anchors = 3 * f_h * f_w;
  const float *output_ptr = (float *) det_pred.data;

  float r_ = scale_params.ratio;
  int dw_ = scale_params.dw;
  int dh_ = scale_params.dh;

  // have c=3 indicate 3 anchors at one grid
  unsigned int count = 0;
  auto &stride_anchors = center_anchors[stride];

  for (unsigned int i = 0; i < num_anchors; ++i)
  {
    const float *row_ptr = output_ptr + i * 16;
    float obj_conf = sigmoid(row_ptr[4]);
    if (obj_conf < score_threshold) continue; // filter first.
    float cls_conf = sigmoid(row_ptr[15]);
    if (cls_conf < score_threshold) continue; // face score.

    int grid0 = stride_anchors.at(i).grid0; // w
    int grid1 = stride_anchors.at(i).grid1; // h
    float anchor_w = stride_anchors.at(i).width;
    float anchor_h = stride_anchors.at(i).height;

    // bounding box
    const float *offsets = row_ptr;
    float dx = sigmoid(offsets[0]);
    float dy = sigmoid(offsets[1]);
    float dw = sigmoid(offsets[2]);
    float dh = sigmoid(offsets[3]);

    float cx = (dx * 2.f - 0.5f + (float) grid0) * (float) stride;
    float cy = (dy * 2.f - 0.5f + (float) grid1) * (float) stride;
    float w = std::pow(dw * 2.f, 2) * anchor_w;
    float h = std::pow(dh * 2.f, 2) * anchor_h;

    types::BoxfWithLandmarks box_kps;
    float x1 = ((cx - w / 2.f) - (float) dw_) / r_;
    float y1 = ((cy - h / 2.f) - (float) dh_) / r_;
    float x2 = ((cx + w / 2.f) - (float) dw_) / r_;
    float y2 = ((cy + h / 2.f) - (float) dh_) / r_;
    box_kps.box.x1 = std::max(0.f, x1);
    box_kps.box.y1 = std::max(0.f, y1);
    box_kps.box.x2 = std::min(img_width - 1.f, x2);
    box_kps.box.y2 = std::min(img_height - 1.f, y2);
    box_kps.box.score = cls_conf;
    box_kps.box.label = 1;
    box_kps.box.label_text = "face";
    box_kps.box.flag = true;

    // landmarks
    const float *kps_offsets = row_ptr + 5;
    for (unsigned int j = 0; j < 10; j += 2)
    {
      float kps_dx = kps_offsets[j];
      float kps_dy = kps_offsets[j + 1];
      float kps_x = (kps_dx * anchor_w + grid0 * (float) stride);
      float kps_y = (kps_dy * anchor_h + grid1 * (float) stride);

      cv::Point2f kps;
      kps_x = (kps_x - (float) dw_) / r_;
      kps_y = (kps_y - (float) dh_) / r_;
      kps.x = std::min(std::max(0.f, kps_x), img_width - 1.f);
      kps.y = std::min(std::max(0.f, kps_y), img_height - 1.f);
      box_kps.landmarks.points.push_back(kps);
    }
    box_kps.landmarks.flag = true;
    box_kps.flag = true;

    bbox_kps_collection.push_back(box_kps);

    count += 1; // limit boxes for nms.
    if (count > max_nms)
      break;
  }

  if (bbox_kps_collection.size() > nms_pre_)
  {
    std::sort(
        bbox_kps_collection.begin(), bbox_kps_collection.end(),
        [](const types::BoxfWithLandmarks &a, const types::BoxfWithLandmarks &b)
        { return a.box.score > b.box.score; }
    ); // sort inplace
    // trunc
    bbox_kps_collection.resize(nms_pre_);
  }
}

void NCNNYOLO5Face::nms_bboxes_kps(std::vector<types::BoxfWithLandmarks> &input,
                                   std::vector<types::BoxfWithLandmarks> &output,
                                   float iou_threshold, unsigned int topk)
{
  if (input.empty()) return;
  std::sort(
      input.begin(), input.end(),
      [](const types::BoxfWithLandmarks &a, const types::BoxfWithLandmarks &b)
      { return a.box.score > b.box.score; }
  );
  const unsigned int box_num = input.size();
  std::vector<int> merged(box_num, 0);

  unsigned int count = 0;
  for (unsigned int i = 0; i < box_num; ++i)
  {
    if (merged[i]) continue;
    std::vector<types::BoxfWithLandmarks> buf;

    buf.push_back(input[i]);
    merged[i] = 1;

    for (unsigned int j = i + 1; j < box_num; ++j)
    {
      if (merged[j]) continue;

      float iou = static_cast<float>(input[i].box.iou_of(input[j].box));

      if (iou > iou_threshold)
      {
        merged[j] = 1;
        buf.push_back(input[j]);
      }

    }
    output.push_back(buf[0]);

    // keep top k
    count += 1;
    if (count >= topk)
      break;
  }
}

void NCNNYOLO5Face::print_debug_string()
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