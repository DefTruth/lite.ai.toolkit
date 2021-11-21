//
// Created by DefTruth on 2021/11/20.
//

#include "ncnn_retinaface.h"
#include "lite/utils.h"

using ncnncv::NCNNRetinaFace;

NCNNRetinaFace::NCNNRetinaFace(const std::string &_param_path,
                               const std::string &_bin_path,
                               unsigned int _num_threads,
                               int _input_height,
                               int _input_width) :
    BasicNCNNHandler(_param_path, _bin_path, _num_threads),
    input_height(_input_height), input_width(_input_width)
{
}

void NCNNRetinaFace::transform(const cv::Mat &mat, ncnn::Mat &in)
{
  cv::Mat mat_rs;
  cv::resize(mat, mat_rs, cv::Size(input_width, input_height));
  in = ncnn::Mat::from_pixels(mat_rs.data, ncnn::Mat::PIXEL_BGR, input_width, input_height);
  in.substract_mean_normalize(mean_vals, norm_vals);
}

void NCNNRetinaFace::detect(const cv::Mat &mat, std::vector<types::Boxf> &detected_boxes,
                            float score_threshold, float iou_threshold,
                            unsigned int topk, unsigned int nms_type)
{
  if (mat.empty()) return;
  float img_height = static_cast<float>(mat.rows);
  float img_width = static_cast<float>(mat.cols);

  // 1. make input tensor
  ncnn::Mat input;
  this->transform(mat, input);
  // 2. inference & extract
  auto extractor = net->create_extractor();
  extractor.set_light_mode(false);  // default
  extractor.set_num_threads(num_threads);
  extractor.input("input", input);
  // 3.rescale & exclude.
  std::vector<types::Boxf> bbox_collection;
  this->generate_bboxes(bbox_collection, extractor, score_threshold, img_height, img_width);
  // 4. hard|blend|offset nms with topk.
  this->nms(bbox_collection, detected_boxes, iou_threshold, topk, nms_type);
}

void NCNNRetinaFace::generate_anchors(const int target_height, const int target_width,
                                      std::vector<RetinaAnchor> &anchors)
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
    auto f_map = feature_maps.at(k); // e.g [640//8,640//8]
    auto tmp_min_sizes = min_sizes.at(k); // e.g [8,16]
    int f_h = f_map.at(0);
    int f_w = f_map.at(1);

    for (int i = 0; i < f_h; ++i)
    {
      for (int j = 0; j < f_w; ++j)
      {
        for (auto min_size: tmp_min_sizes)
        {
          float s_kx = (float) min_size / (float) target_width; // e.g 16/w
          float s_ky = (float) min_size / (float) target_height; // e.g 16/h
          // (x + 0.5) * step / w normalized loc mapping to input width
          // (y + 0.5) * step / h normalized loc mapping to input height
          float cx = ((float) j + 0.5f) * (float) steps.at(k) / (float) target_width;
          float cy = ((float) i + 0.5f) * (float) steps.at(k) / (float) target_height;

          anchors.push_back(RetinaAnchor{cx, cy, s_kx, s_ky}); // without clip
        }
      }
    }
  }
}

void NCNNRetinaFace::generate_bboxes(std::vector<types::Boxf> &bbox_collection,
                                     ncnn::Extractor &extractor, float score_threshold,
                                     float img_height, float img_width)
{
  ncnn::Mat bboxes, probs;
  extractor.extract("bbox", bboxes); // c=1 h=? w=4
  extractor.extract("conf", probs); // c=1 h=? w=2
#ifdef LITENCNN_DEBUG
  BasicNCNNHandler::print_shape(bboxes, "bbox");
  BasicNCNNHandler::print_shape(probs, "conf");
#endif
  const unsigned int bbox_num = bboxes.h; // n = ?

  std::vector<RetinaAnchor> anchors;
  this->generate_anchors(input_height, input_width, anchors);

  const unsigned int num_anchors = anchors.size();
  if (num_anchors != bbox_num)
    throw std::runtime_error("mismatch num_anchors != bbox_num");

  const float *bboxes_ptr = (float *) bboxes.data;
  const float *probs_ptr = (float *) probs.data;

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

#if LITENCNN_DEBUG
  std::cout << "detected num_anchors: " << num_anchors << "\n";
  std::cout << "generate_bboxes num: " << bbox_collection.size() << "\n";
#endif
}

void NCNNRetinaFace::nms(std::vector<types::Boxf> &input, std::vector<types::Boxf> &output,
                         float iou_threshold, unsigned int topk,
                         unsigned int nms_type)
{
  if (nms_type == NMS::BLEND) lite::utils::blending_nms(input, output, iou_threshold, topk);
  else if (nms_type == NMS::OFFSET) lite::utils::offset_nms(input, output, iou_threshold, topk);
  else lite::utils::hard_nms(input, output, iou_threshold, topk);
}















































