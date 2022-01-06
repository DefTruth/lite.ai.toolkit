//
// Created by DefTruth on 2021/12/30.
//

#include "mnn_scrfd.h"
#include "lite/utils.h"

using mnncv::MNNSCRFD;

MNNSCRFD::MNNSCRFD(const std::string &_mnn_path, unsigned int _num_threads) :
    BasicMNNHandler(_mnn_path, _num_threads)
{
  initialize_pretreat();
}

inline void MNNSCRFD::initialize_pretreat()
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

void MNNSCRFD::initial_context()
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

inline void MNNSCRFD::transform(const cv::Mat &mat_rs)
{
  pretreat->convert(mat_rs.data, input_width, input_height, mat_rs.step[0], input_tensor);
}

void MNNSCRFD::resize_unscale(const cv::Mat &mat, cv::Mat &mat_rs,
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

void MNNSCRFD::detect(const cv::Mat &mat, std::vector<types::BoxfWithLandmarks> &detected_boxes_kps,
                      float score_threshold, float iou_threshold, unsigned int topk)
{
  if (mat.empty()) return;
  auto img_height = static_cast<float>(mat.rows);
  auto img_width = static_cast<float>(mat.cols);

  // resize & unscale
  cv::Mat mat_rs;
  SCRFDScaleParams scale_params;
  this->resize_unscale(mat, mat_rs, input_height, input_width, scale_params);

  // 1. make input tensor
  this->transform(mat_rs);

  // 2. inference scores & boxes.
  mnn_interpreter->runSession(mnn_session);
  auto output_tensors = mnn_interpreter->getSessionOutputAll(mnn_session);

  // 3. rescale & exclude.
  std::vector<types::BoxfWithLandmarks> bbox_kps_collection;
  this->generate_bboxes_kps(scale_params, bbox_kps_collection, output_tensors,
                            score_threshold, img_height, img_width);
  // 4. hard nms with topk.
  this->nms_bboxes_kps(bbox_kps_collection, detected_boxes_kps, iou_threshold, topk);
}

void MNNSCRFD::generate_points(const int target_height, const int target_width)
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
      // x
      for (unsigned int j = 0; j < num_grid_w; ++j)
      {
        // num_anchors, col major
        for (unsigned int k = 0; k < num_anchors; ++k)
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

void MNNSCRFD::generate_bboxes_kps(const SCRFDScaleParams &scale_params,
                                   std::vector<types::BoxfWithLandmarks> &bbox_kps_collection,
                                   const std::map<std::string, MNN::Tensor *> &output_tensors,
                                   float score_threshold, float img_height,
                                   float img_width)
{
  // score_8,score_16,score_32,bbox_8,bbox_16,bbox_32
  auto device_score_8 = output_tensors.at("score_8");
  auto device_score_16 = output_tensors.at("score_16");
  auto device_score_32 = output_tensors.at("score_32");
  auto device_bbox_8 = output_tensors.at("bbox_8");
  auto device_bbox_16 = output_tensors.at("bbox_16");
  auto device_bbox_32 = output_tensors.at("bbox_32");
  this->generate_points(input_height, input_width);

  MNN::Tensor host_score_8(device_score_8, device_score_8->getDimensionType());
  MNN::Tensor host_score_16(device_score_16, device_score_16->getDimensionType());
  MNN::Tensor host_score_32(device_score_32, device_score_32->getDimensionType());

}

void MNNSCRFD::generate_bboxes_single_stride(
    const SCRFDScaleParams &scale_params, MNN::Tensor &score_pred, MNN::Tensor &bbox_pred,
    unsigned int stride, float score_threshold, float img_height, float img_width,
    std::vector<types::BoxfWithLandmarks> &bbox_kps_collection)
{

}

void MNNSCRFD::generate_bboxes_kps_single_stride(
    const SCRFDScaleParams &scale_params, MNN::Tensor &score_pred, MNN::Tensor &bbox_pred,
    MNN::Tensor &kps_pred, unsigned int stride, float score_threshold, float img_height,
    float img_width, std::vector<types::BoxfWithLandmarks> &bbox_kps_collection)
{

}

void MNNSCRFD::nms_bboxes_kps(std::vector<types::BoxfWithLandmarks> &input,
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

























