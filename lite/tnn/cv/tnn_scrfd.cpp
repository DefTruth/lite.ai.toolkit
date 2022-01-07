//
// Created by DefTruth on 2021/12/30.
//

#include "tnn_scrfd.h"
#include "lite/utils.h"

using tnncv::TNNSCRFD;

TNNSCRFD::TNNSCRFD(const std::string &_proto_path,
                   const std::string &_model_path,
                   unsigned int _num_threads) :
    BasicTNNHandler(_proto_path, _model_path, _num_threads)
{
  initial_context();
}

void TNNSCRFD::initial_context()
{
  if (num_outputs == 6)
  {
    fmc = 3;
    feat_stride_fpn = {8, 16, 32};
    num_anchors = 2;
    use_kps = false;
  } // kps
  else if (num_outputs == 9)
  {
    fmc = 3;
    feat_stride_fpn = {8, 16, 32};
    num_anchors = 2;
    use_kps = true;
  }

}

void TNNSCRFD::resize_unscale(const cv::Mat &mat, cv::Mat &mat_rs,
                              int target_height, int target_width,
                              SCRFDScaleParams &scale_params)
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
  cv::Mat new_unpad_mat = mat.clone();
  cv::resize(new_unpad_mat, new_unpad_mat, cv::Size(new_unpad_w, new_unpad_h));
  new_unpad_mat.copyTo(mat_rs(cv::Rect(dw, dh, new_unpad_w, new_unpad_h)));

  // record scale params.
  scale_params.ratio = r;
  scale_params.dw = dw;
  scale_params.dh = dh;
  scale_params.flag = true;
}

void TNNSCRFD::transform(const cv::Mat &mat_rs)
{
  // push into input_mat, RGB
  cv::Mat canvas;
  cv::cvtColor(mat_rs, canvas, cv::COLOR_BGR2RGB);
  input_mat = std::make_shared<tnn::Mat>(input_device_type, tnn::N8UC3,
                                         input_shape, (void *) canvas.data);
  if (!input_mat->GetData())
  {
#ifdef LITETNN_DEBUG
    std::cout << "input_mat == nullptr! transform failed\n";
#endif
  }
}

void TNNSCRFD::detect(const cv::Mat &mat, std::vector<types::BoxfWithLandmarks> &detected_boxes_kps,
                      float score_threshold, float iou_threshold, unsigned int topk)
{
  if (mat.empty()) return;
  auto img_height = static_cast<float>(mat.rows);
  auto img_width = static_cast<float>(mat.cols);

  // resize & unscale
  cv::Mat mat_rs;
  SCRFDScaleParams scale_params;
  this->resize_unscale(mat, mat_rs, input_height, input_width, scale_params);

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

  // 3. rescale & exclude.
  std::vector<types::BoxfWithLandmarks> bbox_kps_collection;
  this->generate_bboxes_kps(scale_params, bbox_kps_collection, instance,
                            score_threshold, img_height, img_width);
  // 4. hard nms with topk.
  this->nms_bboxes_kps(bbox_kps_collection, detected_boxes_kps, iou_threshold, topk);
}

void TNNSCRFD::generate_points(const int target_height, const int target_width)
{
  if (center_points_is_update) return;
  // 8, 16, 32
  for (auto stride : feat_stride_fpn)
  {
    unsigned int num_grid_w = target_width / stride;
    unsigned int num_grid_h = target_height / stride;
    // y
    for (unsigned int i = 0; i < num_grid_h; ++i)
    {
      // num_anchors, col major
      for (unsigned int k = 0; k < num_anchors; ++k)
      {
        // x
        for (unsigned int j = 0; j < num_grid_w; ++j)
        {
          SCRFDPoint point;
          point.cx = (float) j;
          point.cy = (float) i;
          point.stride = (float) stride;
          center_points[stride].push_back(point);
        }
      }
    }
  }

  center_points_is_update = true;
}

void TNNSCRFD::generate_bboxes_kps(const SCRFDScaleParams &scale_params,
                                   std::vector<types::BoxfWithLandmarks> &bbox_kps_collection,
                                   std::shared_ptr<tnn::Instance> &_instance,
                                   float score_threshold, float img_height,
                                   float img_width)
{
  tnn::MatConvertParam cvt_param;
  std::shared_ptr<tnn::Mat> score_8, score_16, score_32, bbox_8, bbox_16, bbox_32;
  tnn::Status status_score_8, status_score_16, status_score_32, status_bbox_8, status_bbox_16, status_bbox_32;
  status_score_8 = _instance->GetOutputMat(score_8, cvt_param, "score_8", output_device_type);   // e.g [1,12800,1]
  status_score_16 = _instance->GetOutputMat(score_16, cvt_param, "score_16", output_device_type); // e.g [1,3200,1]
  status_score_32 = _instance->GetOutputMat(score_32, cvt_param, "score_32", output_device_type); // e.g [1,800,1]
  status_bbox_8 = _instance->GetOutputMat(bbox_8, cvt_param, "bbox_8", output_device_type);     // e.g [1,12800,4]
  status_bbox_16 = _instance->GetOutputMat(bbox_16, cvt_param, "bbox_16", output_device_type);   // e.g [1,3200,4]
  status_bbox_32 = _instance->GetOutputMat(bbox_32, cvt_param, "bbox_32", output_device_type);   // e.g [1,800,4]
  this->generate_points(input_height, input_width);

  if (status_score_8 != tnn::TNN_OK || status_score_16 != tnn::TNN_OK || status_score_32 != tnn::TNN_OK ||
      status_bbox_8 != tnn::TNN_OK || status_bbox_16 != tnn::TNN_OK || status_bbox_32 != tnn::TNN_OK)
  {
#ifdef LITETNN_DEBUG
    std::cout << "instance->GetOutputMat failed!:" << status_score_8.description().c_str() << "\n";
    std::cout << "instance->GetOutputMat failed!:" << status_score_16.description().c_str() << "\n";
    std::cout << "instance->GetOutputMat failed!:" << status_score_32.description().c_str() << "\n";
    std::cout << "instance->GetOutputMat failed!:" << status_bbox_8.description().c_str() << "\n";
    std::cout << "instance->GetOutputMat failed!:" << status_bbox_16.description().c_str() << "\n";
    std::cout << "instance->GetOutputMat failed!:" << status_bbox_32.description().c_str() << "\n";
#endif
    return;
  }

  bbox_kps_collection.clear();

  if (use_kps)
  {
    std::shared_ptr<tnn::Mat> kps_8, kps_16, kps_32;
    tnn::Status status_kps_8, status_kps_16, status_kps_32;
    status_kps_8 = _instance->GetOutputMat(kps_8, cvt_param, "kps_8", output_device_type);   // e.g [1,12800,10]
    status_kps_16 = _instance->GetOutputMat(kps_16, cvt_param, "kps_16", output_device_type); // e.g [1,3200,10]
    status_kps_32 = _instance->GetOutputMat(kps_32, cvt_param, "kps_32", output_device_type); // e.g [1,800,10]
    if (status_kps_8 != tnn::TNN_OK || status_kps_16 != tnn::TNN_OK || status_kps_32 != tnn::TNN_OK)
    {
#ifdef LITETNN_DEBUG
      std::cout << "instance->GetOutputMat failed!:" << status_kps_8.description().c_str() << "\n";
      std::cout << "instance->GetOutputMat failed!:" << status_kps_16.description().c_str() << "\n";
      std::cout << "instance->GetOutputMat failed!:" << status_kps_32.description().c_str() << "\n";
#endif
      return;
    }

    // level 8 & 16 & 32 with kps
    this->generate_bboxes_kps_single_stride(scale_params, score_8, bbox_8, kps_8, 8, score_threshold,
                                            img_height, img_width, bbox_kps_collection);
    this->generate_bboxes_kps_single_stride(scale_params, score_16, bbox_16, kps_16, 16, score_threshold,
                                            img_height, img_width, bbox_kps_collection);
    this->generate_bboxes_kps_single_stride(scale_params, score_32, bbox_32, kps_32, 32, score_threshold,
                                            img_height, img_width, bbox_kps_collection);
  } // no kps
  else
  {
    // level 8 & 16 & 32
    this->generate_bboxes_single_stride(scale_params, score_8, bbox_8, 8, score_threshold,
                                        img_height, img_width, bbox_kps_collection);
    this->generate_bboxes_single_stride(scale_params, score_16, bbox_16, 16, score_threshold,
                                        img_height, img_width, bbox_kps_collection);
    this->generate_bboxes_single_stride(scale_params, score_32, bbox_32, 32, score_threshold,
                                        img_height, img_width, bbox_kps_collection);
  }
#if LITETNN_DEBUG
  std::cout << "generate_bboxes_kps num: " << bbox_kps_collection.size() << "\n";
#endif
}


void TNNSCRFD::generate_bboxes_single_stride(
    const SCRFDScaleParams &scale_params, std::shared_ptr<tnn::Mat> &score_pred,
    std::shared_ptr<tnn::Mat> &bbox_pred, unsigned int stride, float score_threshold,
    float img_height, float img_width, std::vector<types::BoxfWithLandmarks> &bbox_kps_collection)
{
  unsigned int nms_pre_ = (stride / 8) * nms_pre; // 1 * 1000,2*1000,...
  nms_pre_ = nms_pre_ >= nms_pre ? nms_pre_ : nms_pre;

  auto stride_dims = score_pred->GetDims();
  const unsigned int num_points = stride_dims.at(1);  // 12800
  const float *score_ptr = (float *) score_pred->GetData();  // [1,12800,1]
  const float *bbox_ptr = (float *) bbox_pred->GetData();    // [1,12800,4]

  float ratio = scale_params.ratio;
  int dw = scale_params.dw;
  int dh = scale_params.dh;

  unsigned int count = 0;
  auto &stride_points = center_points[stride];

  for (unsigned int i = 0; i < num_points; ++i)
  {
    const float cls_conf = score_ptr[i];
    if (cls_conf < score_threshold) continue; // filter
    auto &point = stride_points.at(i);
    const float cx = point.cx; // cx
    const float cy = point.cy; // cy
    const float s = point.stride; // stride

    // bbox
    const float *offsets = bbox_ptr + i * 4;
    float l = offsets[0]; // left
    float t = offsets[1]; // top
    float r = offsets[2]; // right
    float b = offsets[3]; // bottom

    types::BoxfWithLandmarks box_kps;
    float x1 = ((cx - l) * s - (float) dw) / ratio;  // cx - l x1
    float y1 = ((cy - t) * s - (float) dh) / ratio;  // cy - t y1
    float x2 = ((cx + r) * s - (float) dw) / ratio;  // cx + r x2
    float y2 = ((cy + b) * s - (float) dh) / ratio;  // cy + b y2
    box_kps.box.x1 = std::max(0.f, x1);
    box_kps.box.y1 = std::max(0.f, y1);
    box_kps.box.x2 = std::min(img_width, x2);
    box_kps.box.y2 = std::min(img_height, y2);
    box_kps.box.score = cls_conf;
    box_kps.box.label = 1;
    box_kps.box.label_text = "face";
    box_kps.box.flag = true;
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

void TNNSCRFD::generate_bboxes_kps_single_stride(
    const SCRFDScaleParams &scale_params, std::shared_ptr<tnn::Mat> &score_pred,
    std::shared_ptr<tnn::Mat> &bbox_pred, std::shared_ptr<tnn::Mat> &kps_pred,
    unsigned int stride, float score_threshold, float img_height, float img_width,
    std::vector<types::BoxfWithLandmarks> &bbox_kps_collection)
{
  unsigned int nms_pre_ = (stride / 8) * nms_pre; // 1 * 1000,2*1000,...
  nms_pre_ = nms_pre_ >= nms_pre ? nms_pre_ : nms_pre;

  auto stride_dims = score_pred->GetDims();
  const unsigned int num_points = stride_dims.at(1);  // 12800
  const float *score_ptr = (float *) score_pred->GetData();  // [1,12800,1]
  const float *bbox_ptr = (float *) bbox_pred->GetData();    // [1,12800,4]
  const float *kps_ptr = (float *) kps_pred->GetData();      // [1,12800,10]

  float ratio = scale_params.ratio;
  int dw = scale_params.dw;
  int dh = scale_params.dh;

  unsigned int count = 0;
  auto &stride_points = center_points[stride];

  for (unsigned int i = 0; i < num_points; ++i)
  {
    const float cls_conf = score_ptr[i];
    if (cls_conf < score_threshold) continue; // filter
    auto &point = stride_points.at(i);
    const float cx = point.cx; // cx
    const float cy = point.cy; // cy
    const float s = point.stride; // stride

    // bbox
    const float *offsets = bbox_ptr + i * 4;
    float l = offsets[0]; // left
    float t = offsets[1]; // top
    float r = offsets[2]; // right
    float b = offsets[3]; // bottom

    types::BoxfWithLandmarks box_kps;
    float x1 = ((cx - l) * s - (float) dw) / ratio;  // cx - l x1
    float y1 = ((cy - t) * s - (float) dh) / ratio;  // cy - t y1
    float x2 = ((cx + r) * s - (float) dw) / ratio;  // cx + r x2
    float y2 = ((cy + b) * s - (float) dh) / ratio;  // cy + b y2
    box_kps.box.x1 = std::max(0.f, x1);
    box_kps.box.y1 = std::max(0.f, y1);
    box_kps.box.x2 = std::min(img_width, x2);
    box_kps.box.y2 = std::min(img_height, y2);
    box_kps.box.score = cls_conf;
    box_kps.box.label = 1;
    box_kps.box.label_text = "face";
    box_kps.box.flag = true;

    // landmarks
    const float *kps_offsets = kps_ptr + i * 10;
    for (unsigned int j = 0; j < 10; j += 2)
    {
      cv::Point2f kps;
      float kps_l = kps_offsets[j];
      float kps_t = kps_offsets[j + 1];
      float kps_x = ((cx - kps_l) * s - (float) dw) / ratio;  // cx - l x
      float kps_y = ((cy - kps_t) * s - (float) dh) / ratio;  // cy - t y
      kps.x = std::min(std::max(0.f, kps_x), img_width);
      kps.y = std::min(std::max(0.f, kps_y), img_height);
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

void TNNSCRFD::nms_bboxes_kps(std::vector<types::BoxfWithLandmarks> &input,
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








































