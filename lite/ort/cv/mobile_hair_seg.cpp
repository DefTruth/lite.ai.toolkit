//
// Created by DefTruth on 2022/6/22.
//

#include "mobile_hair_seg.h"
#include "lite/ort/core/ort_utils.h"
#include "lite/utils.h"

using ortcv::MobileHairSeg;

Ort::Value MobileHairSeg::transform(const cv::Mat &mat)
{
  cv::Mat canvas;
  cv::resize(mat, canvas, cv::Size(input_node_dims.at(3), input_node_dims.at(2)));
  cv::cvtColor(canvas, canvas, cv::COLOR_BGR2RGB);
  canvas.convertTo(canvas, CV_32FC3, 1.f / 127.5f, -1.f);
  // e.g (1,3,224,224)
  return ortcv::utils::transform::create_tensor(
      canvas, input_node_dims, memory_info_handler,
      input_values_handler, ortcv::utils::transform::CHW); // deepcopy inside
}

void MobileHairSeg::detect(const cv::Mat &mat, types::HairSegContent &content,
                           float score_threshold, bool remove_noise)
{
  if (mat.empty()) return;

  // 1. make input tensor
  Ort::Value input_tensor = this->transform(mat);
  // 2. inference
  auto output_tensors = ort_session->Run(
      Ort::RunOptions{nullptr}, input_node_names.data(),
      &input_tensor, 1, output_node_names.data(), num_outputs
  );
  // 3. generate mask
  this->generate_mask(output_tensors, mat, content, score_threshold, remove_noise);
}

static inline void __zero_if_small_inplace(float *mutable_ptr, float &score)
{ if ((*mutable_ptr) < score) *mutable_ptr = 0.f; }

void MobileHairSeg::generate_mask(std::vector<Ort::Value> &output_tensors, const cv::Mat &mat,
                                  types::HairSegContent &content, float score_threshold,
                                  bool remove_noise)
{
  Ort::Value &output = output_tensors.at(0); // (1,2,h,w) 0~1
  const unsigned int h = mat.rows;
  const unsigned int w = mat.cols;

  auto output_dims = output.GetTypeInfo().GetTensorTypeAndShapeInfo().GetShape();
  const unsigned int out_h = output_dims.at(2);
  const unsigned int out_w = output_dims.at(3);
  const unsigned int element_size = out_h * out_w;

  float *output_ptr = output.GetTensorMutableData<float>() + element_size; // only need positive prob

  // remove small values
  if (score_threshold > 0.001f)
    for (unsigned int i = 0; i < element_size; ++i)
      __zero_if_small_inplace(output_ptr + i, score_threshold);

  cv::Mat mask(out_h, out_w, CV_32FC1, output_ptr);
  // post process
  if (remove_noise) lite::utils::remove_small_connected_area(mask, 0.05f);
  if (out_h != h || out_w != w) cv::resize(mask, mask, cv::Size(w, h));

  content.mask = mask;
  content.flag = true;
}