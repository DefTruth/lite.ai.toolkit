//
// Created by DefTruth on 2021/10/6.
//

#include "mnn_nanodet_efficientnet_lite.h"
#include "mnn/core/mnn_utils.h"

using mnncv::MNNNanoDetEfficientNetLite;

MNNNanoDetEfficientNetLite::MNNNanoDetEfficientNetLite(
    const std::string _mnn_path, unsigned int _num_threads) :
    BasicMNNHandler(_mnn_path, _num_threads)
{
  initialize_pretreat();
}

inline void MNNNanoDetEfficientNetLite::initialize_pretreat()
{
  pretreat = std::shared_ptr<MNN::CV::ImageProcess>(
      MNN::CV::ImageProcess::create(
          MNN::CV::BGR,
          MNN::CV::BGR,
          mean_vals, 3,
          norm_vals, 3
      )
  );
}

inline void MNNNanoDetEfficientNetLite::transform(const cv::Mat &mat_rs)
{
  pretreat->convert(mat_rs.data, input_width, input_height, mat_rs.step[0], input_tensor);
}


void MNNNanoDetEfficientNetLite::resize_unscale(const cv::Mat &mat, cv::Mat &mat_rs,
                                                int target_height, int target_width,
                                                NanoLiteScaleParams &scale_params)
{
  if (mat.empty()) return;
  int img_height = static_cast<int>(mat.rows);
  int img_width = static_cast<int>(mat.cols);

  mat_rs = cv::Mat(target_height, target_width, CV_8UC3,
                   cv::Scalar(0, 0, 0));
  // scale ratio (new / old) new_shape(h,w)
  float w_r = (float) target_width / (float) img_width;
  float h_r = (float) target_height / (float) img_height;
  float r = std::fmin(w_r, h_r);
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

void MNNNanoDetEfficientNetLite::detect(
    const cv::Mat &mat, std::vector<types::Boxf> &detected_boxes,
    float score_threshold, float iou_threshold,
    unsigned int topk, unsigned int nms_type)
{
  if (mat.empty()) return;
  auto img_height = static_cast<float>(mat.rows);
  auto img_width = static_cast<float>(mat.cols);

  // resize & unscale
  cv::Mat mat_rs;
  NanoLiteScaleParams scale_params;
  this->resize_unscale(mat, mat_rs, input_height, input_width, scale_params);

  // 1. make input tensor
  this->transform(mat_rs);
  // 2. inference scores & boxes.
  mnn_interpreter->runSession(mnn_session);
  auto output_tensors = mnn_interpreter->getSessionOutputAll(mnn_session);
  // 3. rescale & exclude.
  std::vector<types::Boxf> bbox_collection;
  this->generate_bboxes(scale_params, bbox_collection, output_tensors, score_threshold, img_height, img_width);
  // 4. hard|blend|offset nms with topk.
  this->nms(bbox_collection, detected_boxes, iou_threshold, topk, nms_type);
}

void MNNNanoDetEfficientNetLite::generate_points(unsigned int target_height, unsigned int target_width)
{
  if (center_points_is_update) return;

  for (auto stride : strides)
  {
    unsigned int num_grid_w = target_width / stride;
    unsigned int num_grid_h = target_height / stride;
    std::vector<NanoLiteCenterPoint> points;

    for (unsigned int g1 = 0; g1 < num_grid_h; ++g1)
    {
      for (unsigned int g0 = 0; g0 < num_grid_w; ++g0)
      {
        float grid0 = (float) g0 + 0.5f;
        float grid1 = (float) g1 + 0.5f;
#ifdef LITE_WIN32
        NanoLiteCenterPoint point;
        point.grid0 = grid0;
        point.grid1 = grid1;
        point.stride = (float) stride;
        points.push_back(point);
#else
        points.push_back((NanoLiteCenterPoint) {grid0, grid1, (float) stride});
#endif
      }
    }
    center_points[stride] = points;
  }

  center_points_is_update = true;
}

void MNNNanoDetEfficientNetLite::generate_bboxes(const NanoLiteScaleParams &scale_params,
                                                 std::vector<types::Boxf> &bbox_collection,
                                                 const std::map<std::string, MNN::Tensor *> &output_tensors,
                                                 float score_threshold, float img_height,
                                                 float img_width)
{
  // device tensor
  auto cls_pred_stride_8 = output_tensors.at("cls_pred_stride_8");   // e.g (1,1600,80)
  auto cls_pred_stride_16 = output_tensors.at("cls_pred_stride_16"); // e.g (1,400,80)
  auto cls_pred_stride_32 = output_tensors.at("cls_pred_stride_32"); // e.g (1,100,80)
  auto dis_pred_stride_8 = output_tensors.at("dis_pred_stride_8");   // e.g (1,1600,4) xyxy (l,t,r,b)
  auto dis_pred_stride_16 = output_tensors.at("dis_pred_stride_16"); // e.g (1,400,4)  xyxy (l,t,r,b)
  auto dis_pred_stride_32 = output_tensors.at("dis_pred_stride_32"); // e.g (1,100,4)  xyxy (l,t,r,b)
  this->generate_points(input_height, input_width); // e.g 320 320

  bbox_collection.clear();
  // level 8 & 16 & 32
  this->generate_bboxes_single_stride(scale_params, cls_pred_stride_8, dis_pred_stride_8, 8,
                                      score_threshold, img_height, img_width, bbox_collection);
  this->generate_bboxes_single_stride(scale_params, cls_pred_stride_16, dis_pred_stride_16, 16,
                                      score_threshold, img_height, img_width, bbox_collection);
  this->generate_bboxes_single_stride(scale_params, cls_pred_stride_32, dis_pred_stride_32, 32,
                                      score_threshold, img_height, img_width, bbox_collection);
#if LITEMNN_DEBUG
  std::cout << "generate_bboxes num: " << bbox_collection.size() << "\n";
#endif
}


void MNNNanoDetEfficientNetLite::generate_bboxes_single_stride(const NanoLiteScaleParams &scale_params,
                                                               const MNN::Tensor *device_cls_pred,
                                                               const MNN::Tensor *device_dis_pred,
                                                               unsigned int stride,
                                                               float score_threshold,
                                                               float img_height,
                                                               float img_width,
                                                               std::vector<types::Boxf> &bbox_collection)
{
  unsigned int nms_pre_ = (stride / 8) * nms_pre; // 1 * 1000,2*1000,...
  nms_pre_ = nms_pre_ >= nms_pre ? nms_pre_ : nms_pre;
  MNN::Tensor host_cls_pred(device_cls_pred, device_cls_pred->getDimensionType()); // e.g (1,1600,80)
  MNN::Tensor host_dis_pred(device_dis_pred, device_dis_pred->getDimensionType()); // e.g (1,1600,4)
  device_cls_pred->copyToHostTensor(&host_cls_pred);
  device_dis_pred->copyToHostTensor(&host_dis_pred);

  auto cls_pred_dims = host_cls_pred.shape(); // e.g (1,1600,80)
  const unsigned int num_points = cls_pred_dims.at(1); // e.g 1600
  const unsigned int num_classes = cls_pred_dims.at(2); // e.g 80

  float ratio = scale_params.ratio;
  int dw = scale_params.dw;
  int dh = scale_params.dh;

  unsigned int count = 0;
  auto &stride_points = center_points[stride];
  for (unsigned int i = 0; i < num_points; ++i)
  {
    const float *scores = host_cls_pred.host<float>() + (i * num_classes); // row ptr
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

    auto &point = stride_points.at(i);
    const float cx = point.grid0; // cx
    const float cy = point.grid1; // cy
    const float s = point.stride; // stride

    const float *offsets = host_dis_pred.host<float>() + (i * 4);

    float l = offsets[0]; // left
    float t = offsets[1]; // top
    float r = offsets[2]; // right
    float b = offsets[3]; // bottom

    types::Boxf box;
    float x1 = ((cx - l) * s - (float) dw) / ratio;  // cx - l x1
    float y1 = ((cy - t) * s - (float) dh) / ratio;  // cy - t y1
    float x2 = ((cx + r) * s - (float) dw) / ratio;  // cx + r x2
    float y2 = ((cy + b) * s - (float) dh) / ratio;  // cy + b y2
    box.x1 = std::fmax(0.f, x1);
    box.y1 = std::fmax(0.f, y1);
    box.x2 = std::fmin(img_width, x2);
    box.y2 = std::fmin(img_height, y2);
    box.score = cls_conf;
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

void MNNNanoDetEfficientNetLite::nms(
    std::vector<types::Boxf> &input, std::vector<types::Boxf> &output,
    float iou_threshold, unsigned int topk, unsigned int nms_type)
{
  if (nms_type == NMS::BLEND) mnncv::utils::blending_nms(input, output, iou_threshold, topk);
  else if (nms_type == NMS::OFFSET) mnncv::utils::offset_nms(input, output, iou_threshold, topk);
  else mnncv::utils::hard_nms(input, output, iou_threshold, topk);
}
