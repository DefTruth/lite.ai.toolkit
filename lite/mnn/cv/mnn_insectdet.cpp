//
// Created by DefTruth on 2022/3/27.
//

#include "mnn_insectdet.h"
#include "lite/utils.h"

using mnncv::MNNInsectDet;

MNNInsectDet::MNNInsectDet(const std::string &_mnn_path, unsigned int _num_threads) :
    BasicMNNHandler(_mnn_path, _num_threads)
{
  initialize_pretreat();
}

inline void MNNInsectDet::initialize_pretreat()
{
  pretreat = std::shared_ptr<MNN::CV::ImageProcess>(
      MNN::CV::ImageProcess::create(
          MNN::CV::BGR,
          MNN::CV::RGB,
          mean_vals, 3,
          norm_vals, 3
      )
  );
}

inline void MNNInsectDet::transform(const cv::Mat &mat_rs)
{
  pretreat->convert(mat_rs.data, input_width, input_height, mat_rs.step[0], input_tensor);
}

void MNNInsectDet::resize_unscale(const cv::Mat &mat, cv::Mat &mat_rs,
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

void MNNInsectDet::detect(const cv::Mat &mat, std::vector<types::Boxf> &detected_boxes,
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
  this->transform(mat_rs);
  // 2. inference scores & boxes.
  mnn_interpreter->runSession(mnn_session);
  auto output_tensors = mnn_interpreter->getSessionOutputAll(mnn_session);
  // 3. rescale & exclude.
  std::vector<types::Boxf> bbox_collection;
  this->generate_bboxes(scale_params, bbox_collection, output_tensors, score_threshold, img_height, img_width);
  // 4. hard|blend|offset nms with topk.
  this->nms(bbox_collection, detected_boxes, iou_threshold, topk);
}

void MNNInsectDet::generate_bboxes(const InsectDetScaleParams &scale_params,
                                   std::vector<types::Boxf> &bbox_collection,
                                   const std::map<std::string, MNN::Tensor *> &output_tensors,
                                   float score_threshold, int img_height,
                                   int img_width)
{
  auto device_output_pred = output_tensors.at("output");
  MNN::Tensor host_output_pred(device_output_pred, device_output_pred->getDimensionType());
  device_output_pred->copyToHostTensor(&host_output_pred);

  auto output_dims = host_output_pred.shape(); // (1,n,6)
  const unsigned int num_anchors = output_dims.at(1); // n = ?
  const float *output_ptr = host_output_pred.host<float>();

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

#if LITEMNN_DEBUG
  std::cout << "generate_bboxes num: " << bbox_collection.size() << "\n";
#endif
}

void MNNInsectDet::nms(std::vector<types::Boxf> &input, std::vector<types::Boxf> &output,
                       float iou_threshold, unsigned int topk)
{
  lite::utils::hard_nms(input, output, iou_threshold, topk);
}