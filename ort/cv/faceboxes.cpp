//
// Created by DefTruth on 2021/8/1.
//

#include "faceboxes.h"
#include "ort/core/ort_utils.h"
#include "lite/utils.h"

using ortcv::FaceBoxes;

Ort::Value FaceBoxes::transform(const cv::Mat &mat)
{
  cv::Mat canva = mat.clone();
  cv::resize(canva, canva, cv::Size(input_node_dims.at(3),
                                    input_node_dims.at(2)));
  // e.g (1,3,640,640) 1xCXHXW
  ortcv::utils::transform::normalize_inplace(canva, mean_vals, scale_vals); // float32
  return ortcv::utils::transform::create_tensor(
      canva, input_node_dims, memory_info_handler,
      input_values_handler, ortcv::utils::transform::CHW);
}

void FaceBoxes::detect(const cv::Mat &mat, std::vector<types::Boxf> &detected_boxes,
                       float score_threshold, float iou_threshold,
                       unsigned int topk, unsigned int nms_type)
{
  if (mat.empty()) return;
  float img_height = static_cast<float>(mat.rows);
  float img_width = static_cast<float>(mat.cols);

  // 1. make input tensor
  Ort::Value input_tensor = this->transform(mat);
  // 2. inference scores & boxes.
  auto output_tensors = ort_session->Run(
      Ort::RunOptions{nullptr}, input_node_names.data(),
      &input_tensor, 1, output_node_names.data(), num_outputs
  );
  // 3. rescale & exclude.
  std::vector<types::Boxf> bbox_collection;
  this->generate_bboxes(bbox_collection, output_tensors, score_threshold, img_height, img_width);
  // 4. hard|blend|offset nms with topk.
  this->nms(bbox_collection, detected_boxes, iou_threshold, topk, nms_type);
}

// ref: https://github.com/zisianw/FaceBoxes.PyTorch/blob/master/layers/functions/prior_box.py
void FaceBoxes::generate_anchors(const int target_height,
                                 const int target_width,
                                 std::vector<FaceBoxesAnchor> &anchors)
{
  std::vector<std::vector<int>> feature_maps;
  for (auto step: steps)
  {
    feature_maps.push_back(
        {
            (int) std::ceilf((float) target_height / (float) step),
            (int) std::ceilf((float) target_width / (float) step)
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
                anchors.push_back((FaceBoxesAnchor) {cx, cy, s_kx, s_ky}); // without clip
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
                anchors.push_back((FaceBoxesAnchor) {cx, cy, s_kx, s_ky}); // without clip
              }
            }

          } // other anchor size
          else
          {
            float cx = ((float) j + 0.5f) * (float) steps.at(k) / (float) target_width;
            float cy = ((float) i + 0.5f) * (float) steps.at(k) / (float) target_height;
            anchors.push_back((FaceBoxesAnchor) {cx, cy, s_kx, s_ky}); // without clip
          }
        }
      }
    }
  }
}


void FaceBoxes::generate_bboxes(std::vector<types::Boxf> &bbox_collection,
                                std::vector<Ort::Value> &output_tensors,
                                float score_threshold,
                                float img_height, float img_width)
{
  Ort::Value &bboxes = output_tensors.at(0); // e.g (1,n,4)
  Ort::Value &probs = output_tensors.at(1); // e.g (1,n,2) after softmax
  auto bbox_dims = output_node_dims.at(0); // (1,n,4)
  const unsigned int bbox_num = bbox_dims.at(1); // n = ?
  const float input_height = static_cast<float>(input_node_dims.at(2)); // e.g 640
  const float input_width = static_cast<float>(input_node_dims.at(3)); // e.g 640

  std::vector<FaceBoxesAnchor> anchors;
  this->generate_anchors(input_height, input_width, anchors);

  const unsigned int num_anchors = anchors.size();

  if (num_anchors != bbox_num)
  {
    std::cout << "num_anchors=" << num_anchors << " but detected bbox_num="
              << bbox_num << std::endl;
    throw std::runtime_error("mismatch num_anchors != bbox_num");
  }

  bbox_collection.clear();
  unsigned int count = 0;
  for (unsigned int i = 0; i < num_anchors; ++i)
  {
    float conf = probs.At<float>({0, i, 1});
    if (conf < score_threshold) continue; // filter first.

    float prior_cx = anchors.at(i).cx;
    float prior_cy = anchors.at(i).cy;
    float prior_s_kx = anchors.at(i).s_kx;
    float prior_s_ky = anchors.at(i).s_ky;

    float dx = bboxes.At<float>({0, i, 0});
    float dy = bboxes.At<float>({0, i, 1});
    float dw = bboxes.At<float>({0, i, 2});
    float dh = bboxes.At<float>({0, i, 3});

    // ref: https://github.com/zisianw/FaceBoxes.PyTorch/blob/master/utils/box_utils.py
    float cx = prior_cx + dx * variance[0] * prior_s_kx;
    float cy = prior_cy + dy * variance[0] * prior_s_ky;
    float w = prior_s_kx * std::expf(dw * variance[1]);
    float h = prior_s_ky * std::expf(dh * variance[1]); // norm coor (0.,1.)

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
#if LITEORT_DEBUG
  std::cout << "detected num_anchors: " << num_anchors << "\n";
  std::cout << "generate_bboxes num: " << bbox_collection.size() << "\n";
#endif
}


void FaceBoxes::nms(std::vector<types::Boxf> &input, std::vector<types::Boxf> &output,
                    float iou_threshold, unsigned int topk, unsigned int nms_type)
{
  if (nms_type == NMS::BLEND) lite::utils::blending_nms(input, output, iou_threshold, topk);
  else if (nms_type == NMS::OFFSET) lite::utils::offset_nms(input, output, iou_threshold, topk);
  else lite::utils::hard_nms(input, output, iou_threshold, topk);
}