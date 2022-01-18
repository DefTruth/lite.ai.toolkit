//
// Created by DefTruth on 2022/1/16.
//

#include "tnn_yolo5face.h"

using tnncv::TNNYOLO5Face;

TNNYOLO5Face::TNNYOLO5Face(const std::string &_proto_path,
                           const std::string &_model_path,
                           unsigned int _num_threads) :
    BasicTNNHandler(_proto_path, _model_path, _num_threads)
{
}

void TNNYOLO5Face::resize_unscale(const cv::Mat &mat, cv::Mat &mat_rs,
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
  cv::Mat new_unpad_mat = mat.clone();
  cv::resize(new_unpad_mat, new_unpad_mat, cv::Size(new_unpad_w, new_unpad_h));
  new_unpad_mat.copyTo(mat_rs(cv::Rect(dw, dh, new_unpad_w, new_unpad_h)));

  // record scale params.
  scale_params.ratio = r;
  scale_params.dw = dw;
  scale_params.dh = dh;
  scale_params.flag = true;
}

void TNNYOLO5Face::transform(const cv::Mat &mat_rs)
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

void TNNYOLO5Face::detect(const cv::Mat &mat, std::vector<types::BoxfWithLandmarks> &detected_boxes_kps,
                          float score_threshold, float iou_threshold, unsigned int topk)
{
  if (mat.empty()) return;
  auto img_height = static_cast<float>(mat.rows);
  auto img_width = static_cast<float>(mat.cols);

  // resize & unscale
  cv::Mat mat_rs;
  YOLO5FaceScaleParams scale_params;
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

void TNNYOLO5Face::generate_bboxes_kps(const YOLO5FaceScaleParams &scale_params,
                                       std::vector<types::BoxfWithLandmarks> &bbox_kps_collection,
                                       std::shared_ptr<tnn::Instance> &_instance,
                                       float score_threshold, float img_height,
                                       float img_width)
{
  tnn::MatConvertParam cvt_param;
  std::shared_ptr<tnn::Mat> output;
  tnn::Status status;

  status = _instance->GetOutputMat(output, cvt_param, "output", output_device_type); // [1,N,16]
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

  bbox_kps_collection.clear();
  unsigned int count = 0;
  for (unsigned int i = 0; i < num_anchors; ++i)
  {
    const float *row_ptr = output_ptr + i * 16;
    float obj_conf = row_ptr[4];
    if (obj_conf < score_threshold) continue; // filter first.
    float cls_conf = row_ptr[15];
    if (cls_conf < score_threshold) continue; // face score.

    // bounding box
    const float *offsets = row_ptr;
    float cx = offsets[0];
    float cy = offsets[1];
    float w = offsets[2];
    float h = offsets[3];

    types::BoxfWithLandmarks box_kps;
    float x1 = ((cx - w / 2.f) - (float) dw_) / r_;
    float y1 = ((cy - h / 2.f) - (float) dh_) / r_;
    float x2 = ((cx + w / 2.f) - (float) dw_) / r_;
    float y2 = ((cy + h / 2.f) - (float) dh_) / r_;
    box_kps.box.x1 = std::max(0.f, x1);
    box_kps.box.y1 = std::max(0.f, y1);
    box_kps.box.x2 = std::min(img_width, x2);
    box_kps.box.y2 = std::min(img_height, y2);
    box_kps.box.score = cls_conf;
    box_kps.box.label = 1;
    box_kps.box.label_text = "face";
    box_kps.box.flag = true;

    // landmarks
    const float *kps_offsets = row_ptr + 5;
    for (unsigned int j = 0; j < 10; j += 2)
    {
      cv::Point2f kps;
      float kps_x = (kps_offsets[j] - (float) dw_) / r_;
      float kps_y = (kps_offsets[j + 1] - (float) dh_) / r_;
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

#if LITETNN_DEBUG
  std::cout << "generate_bboxes_kps num: " << bbox_kps_collection.size() << "\n";
#endif
}

void TNNYOLO5Face::nms_bboxes_kps(std::vector<types::BoxfWithLandmarks> &input,
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