//
// Created by DefTruth on 2021/11/20.
//

#include "mnn_ultraface.h"
#include "lite/utils.h"

using mnncv::MNNUltraFace;

MNNUltraFace::MNNUltraFace(const std::string &_mnn_path, unsigned int _num_threads) :
    BasicMNNHandler(_mnn_path, _num_threads)
{
  initialize_pretreat();
}

inline void MNNUltraFace::initialize_pretreat()
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

void MNNUltraFace::transform(const cv::Mat &mat)
{
  cv::Mat mat_rs;
  cv::resize(mat, mat_rs, cv::Size(input_width, input_height));
  // normalize & HWC -> CHW & BGR -> RGB
  pretreat->convert(mat_rs.data, input_width, input_height, mat_rs.step[0], input_tensor);
}

void MNNUltraFace::detect(const cv::Mat &mat, std::vector<types::Boxf> &detected_boxes,
                          float score_threshold, float iou_threshold,
                          unsigned int topk, unsigned int nms_type)
{
  if (mat.empty()) return;
  // this->transform(mat);
  float img_height = static_cast<float>(mat.rows);
  float img_width = static_cast<float>(mat.cols);

  // 1. make input tensor
  this->transform(mat);
  // 2. inference scores & boxes.
  mnn_interpreter->runSession(mnn_session);
  auto output_tensors = mnn_interpreter->getSessionOutputAll(mnn_session);
  // 3. rescale & exclude.
  std::vector<types::Boxf> bbox_collection;
  this->generate_bboxes(bbox_collection, output_tensors, score_threshold, img_height, img_width);
  // 4. hard|blend nms with topk.
  this->nms(bbox_collection, detected_boxes, iou_threshold, topk, nms_type);
}

void MNNUltraFace::generate_bboxes(std::vector<types::Boxf> &bbox_collection,
                                   const std::map<std::string, MNN::Tensor *> &output_tensors,
                                   float score_threshold, float img_height,
                                   float img_width)
{
  auto device_scores_ptr = output_tensors.at("scores"); // (1,n,2)
  auto device_boxes_ptr = output_tensors.at("boxes");  // (1,n,4)
  MNN::Tensor host_scores_tensor(device_scores_ptr, device_scores_ptr->getDimensionType());
  MNN::Tensor host_boxes_tensor(device_boxes_ptr, device_boxes_ptr->getDimensionType());
  device_scores_ptr->copyToHostTensor(&host_scores_tensor);
  device_boxes_ptr->copyToHostTensor(&host_boxes_tensor);

  auto scores_dims = host_scores_tensor.shape();  // (1,n,2)
  const unsigned int num_anchors = scores_dims.at(1); // n = 17640 (640x480)
  const float *scores_ptr = host_scores_tensor.host<float>();
  const float *boxes_ptr = host_boxes_tensor.host<float>();

  bbox_collection.clear();
  unsigned int count = 0;
  for (unsigned int i = 0; i < num_anchors; ++i)
  {
    float confidence = scores_ptr[2 * i + 1];
    if (confidence < score_threshold) continue;
    types::Boxf box;
    box.x1 = boxes_ptr[4 * i + 0] * img_width;
    box.y1 = boxes_ptr[4 * i + 1] * img_height;
    box.x2 = boxes_ptr[4 * i + 2] * img_width;
    box.y2 = boxes_ptr[4 * i + 3] * img_height;
    box.score = confidence;
    box.label_text = "face";
    box.label = 1;
    box.flag = true;
    bbox_collection.push_back(box);

    count += 1; // limit boxes for nms.
    if (count > max_nms)
      break;
  }
#if LITEMNN_DEBUG
  std::cout << "detected num_anchors: " << num_anchors << "\n";
  std::cout << "generate_bboxes num: " << bbox_collection.size() << "\n";
#endif
}

void MNNUltraFace::nms(std::vector<types::Boxf> &input, std::vector<types::Boxf> &output,
                       float iou_threshold, unsigned int topk,
                       unsigned int nms_type)
{
  if (nms_type == NMS::BLEND) lite::utils::blending_nms(input, output, iou_threshold, topk);
  else if (nms_type == NMS::OFFSET) lite::utils::offset_nms(input, output, iou_threshold, topk);
  else lite::utils::hard_nms(input, output, iou_threshold, topk);
}




























