//
// Created by DefTruth on 2021/11/20.
//

#include "tnn_faceboxes.h"
#include "lite/utils.h"

using tnncv::TNNFaceBoxes;

TNNFaceBoxes::TNNFaceBoxes(const std::string &_proto_path,
                           const std::string &_model_path,
                           unsigned int _num_threads) :
    BasicTNNHandler(_proto_path, _model_path, _num_threads)
{
}

void TNNFaceBoxes::transform(const cv::Mat &mat_rs)
{
  // push into input_mat
  input_mat = std::make_shared<tnn::Mat>(input_device_type, tnn::N8UC3,
                                         input_shape, (void *) mat_rs.data);
  if (!input_mat->GetData())
  {
#ifdef LITETNN_DEBUG
    std::cout << "input_mat == nullptr! transform failed\n";
#endif
  }
}

void TNNFaceBoxes::detect(const cv::Mat &mat, std::vector<types::Boxf> &detected_boxes,
                          float score_threshold, float iou_threshold,
                          unsigned int topk, unsigned int nms_type)
{
  if (mat.empty()) return;
  float img_height = static_cast<float>(mat.rows);
  float img_width = static_cast<float>(mat.cols);

  // 1. make input mat
  cv::Mat mat_rs;
  cv::resize(mat, mat_rs, cv::Size(input_width, input_height));
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
    std::cout << status.description().c_str() << "\n";
#endif
    return;
  }
  // 3. forward
  status = instance->Forward();
  if (status != tnn::TNN_OK)
  {
#ifdef LITETNN_DEBUG
    std::cout << status.description().c_str() << "\n";
#endif
    return;
  }
  // 4. rescale & exclude.
  std::vector<types::Boxf> bbox_collection;
  this->generate_bboxes(bbox_collection, instance, score_threshold, img_height, img_width);
  // 5. hard|blend nms with topk.
  this->nms(bbox_collection, detected_boxes, iou_threshold, topk, nms_type);
}

void TNNFaceBoxes::generate_anchors(const int target_height, const int target_width,
                                    std::vector<FaceBoxesAnchor> &anchors)
{
  std::vector<std::vector<int>> feature_maps;
  for (auto step: steps)
  {
    feature_maps.push_back(
        {
            (int) std::ceil((float) target_height / (float) step),
            (int) std::ceil((float) target_width / (float) step)
        } // ceil
    );
  }

  anchors.clear();
  const int num_feature_map = feature_maps.size();

  for (int k = 0; k < num_feature_map; ++k)
  {
    auto f_map = feature_maps.at(k); // e.g [640//32,640/32]
    auto tmp_min_sizes = min_sizes.at(k); // e.g [32,64,128]
    int f_h = f_map.at(0);
    int f_w = f_map.at(1);
    std::vector<float> offset_32 = {0.f, 0.25f, 0.5f, 0.75f};
    std::vector<float> offset_64 = {0.f, 0.5f};

    for (int i = 0; i < f_h; ++i)
    {
      for (int j = 0; j < f_w; ++j)
      {
        for (auto min_size: tmp_min_sizes)
        {
          float s_kx = (float) min_size / (float) target_width; // e.g 32/w
          float s_ky = (float) min_size / (float) target_height; // e.g 32/h

          // 32 anchor size
          if (min_size == 32)
          {
            // range y offsets first and then x
            for (auto offset_y: offset_32)
            {
              for (auto offset_x: offset_32)
              {
                // (x or y + offset) * step / w or h normalized loc mapping to input size.
                float cx = ((float) j + offset_x) * (float) steps.at(k) / (float) target_width;
                float cy = ((float) i + offset_y) * (float) steps.at(k) / (float) target_height;
                anchors.push_back(FaceBoxesAnchor{cx, cy, s_kx, s_ky}); // without clip
              }
            }

          } // 64 anchor size
          else if (min_size == 64)
          {
            // range y offsets first and then x
            for (auto offset_y: offset_64)
            {
              for (auto offset_x: offset_64)
              {
                float cx = ((float) j + offset_x) * (float) steps.at(k) / (float) target_width;
                float cy = ((float) i + offset_y) * (float) steps.at(k) / (float) target_height;
                anchors.push_back(FaceBoxesAnchor{cx, cy, s_kx, s_ky}); // without clip
              }
            }

          } // other anchor size
          else
          {
            float cx = ((float) j + 0.5f) * (float) steps.at(k) / (float) target_width;
            float cy = ((float) i + 0.5f) * (float) steps.at(k) / (float) target_height;
            anchors.push_back(FaceBoxesAnchor{cx, cy, s_kx, s_ky}); // without clip
          }
        }
      }
    }
  }
}

void TNNFaceBoxes::generate_bboxes(std::vector<types::Boxf> &bbox_collection,
                                   std::shared_ptr<tnn::Instance> &_instance,
                                   float score_threshold, float img_height,
                                   float img_width)
{
  std::shared_ptr<tnn::Mat> bboxes; // (1,n,4)
  std::shared_ptr<tnn::Mat> probs; // (1,n,2)
  tnn::MatConvertParam cvt_param;
  tnn::Status status_bboxes;
  tnn::Status status_probs;

  status_bboxes = _instance->GetOutputMat(bboxes, cvt_param, "bbox", output_device_type);
  status_probs = _instance->GetOutputMat(probs, cvt_param, "conf", output_device_type);

  if (status_bboxes != tnn::TNN_OK || status_probs != tnn::TNN_OK)
  {
#ifdef LITETNN_DEBUG
    std::cout << status_bboxes.description().c_str() << ": "
              << status_probs.description().c_str() << "\n";
#endif
    return;
  }
  auto bbox_dims = bboxes->GetDims();
  const unsigned int bbox_num = bbox_dims.at(1); // n = ?

  std::vector<FaceBoxesAnchor> anchors;
  this->generate_anchors(input_height, input_width, anchors);

  const unsigned int num_anchors = anchors.size();
  if (num_anchors != bbox_num)
    throw std::runtime_error("mismatch num_anchors != bbox_num");

  const float *bboxes_ptr = (float *) bboxes->GetData();
  const float *probs_ptr = (float *) probs->GetData();

  bbox_collection.clear();
  unsigned int count = 0;
  for (unsigned int i = 0; i < num_anchors; ++i)
  {
    float conf = probs_ptr[2 * i + 1];
    if (conf < score_threshold) continue; // filter first.

    float prior_cx = anchors.at(i).cx;
    float prior_cy = anchors.at(i).cy;
    float prior_s_kx = anchors.at(i).s_kx;
    float prior_s_ky = anchors.at(i).s_ky;

    float dx = bboxes_ptr[4 * i + 0];
    float dy = bboxes_ptr[4 * i + 1];
    float dw = bboxes_ptr[4 * i + 2];
    float dh = bboxes_ptr[4 * i + 3];
    // ref: https://github.com/biubug6/Pytorch_Retinaface/blob/master/utils/box_utils.py
    float cx = prior_cx + dx * variance[0] * prior_s_kx;
    float cy = prior_cy + dy * variance[0] * prior_s_ky;
    float w = prior_s_kx * std::exp(dw * variance[1]);
    float h = prior_s_ky * std::exp(dh * variance[1]); // norm coor (0.,1.)

    types::Boxf box;
    box.x1 = (cx - w / 2.f) * img_width;
    box.y1 = (cy - h / 2.f) * img_height;
    box.x2 = (cx + w / 2.f) * img_width;
    box.y2 = (cy + h / 2.f) * img_height;
    box.score = conf;
    box.label = 1;
    box.label_text = "face";
    box.flag = true;
    bbox_collection.push_back(box);

    count += 1; // limit boxes for nms.
    if (count > max_nms)
      break;
  }
#if LITETNN_DEBUG
  std::cout << "detected num_anchors: " << num_anchors << "\n";
  std::cout << "generate_bboxes num: " << bbox_collection.size() << "\n";
#endif
}

void TNNFaceBoxes::nms(std::vector<types::Boxf> &input, std::vector<types::Boxf> &output,
                       float iou_threshold, unsigned int topk,
                       unsigned int nms_type)
{
  if (nms_type == NMS::BLEND) lite::utils::blending_nms(input, output, iou_threshold, topk);
  else if (nms_type == NMS::OFFSET) lite::utils::offset_nms(input, output, iou_threshold, topk);
  else lite::utils::hard_nms(input, output, iou_threshold, topk);
}












































