//
// Created by DefTruth on 2022/6/19.
//

#include "face_hair_seg.h"
#include "lite/ort/core/ort_utils.h"
#include "lite/utils.h"

using ortcv::FaceHairSeg;

Ort::Value FaceHairSeg::transform(const cv::Mat &mat)
{
  cv::Mat canvas;
  cv::resize(mat, canvas, cv::Size(input_node_dims.at(3), input_node_dims.at(2)));
  cv::cvtColor(canvas, canvas, cv::COLOR_BGR2RGB);
  canvas.convertTo(canvas, CV_32FC3, 1.f / 255.f, 0.f);
  // e.g (1,3,224,224)
  return ortcv::utils::transform::create_tensor(
      canvas, input_node_dims, memory_info_handler,
      input_values_handler, ortcv::utils::transform::CHW); // deepcopy inside
}

void FaceHairSeg::detect(const cv::Mat &mat, types::FaceHairSegContent &content,
                         bool remove_noise)
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
  this->generate_mask(output_tensors, mat, content, remove_noise);
}

static inline float argmax(float *mutable_ptr, const unsigned int &step)
{
  std::vector<float> logits(3, 0.f);
  logits[0] = *mutable_ptr; // background
  logits[1] = *(mutable_ptr + step); // face
  logits[2] = *(mutable_ptr + 2 * step); // hair
  float label = 0.f;
  float max_logit = logits[0];
  for (unsigned int i = 1; i < 3; ++i)
  {
    if (logits[i] > max_logit)
    {
      max_logit = logits[i];
      label = (float) i;
    }
  }
  // normalize -> 0.~1.
  return label / 2.f; // 0. bgr 0.5 face 1. hair
}

void FaceHairSeg::generate_mask(std::vector<Ort::Value> &output_tensors, const cv::Mat &mat,
                                types::FaceHairSegContent &content,
                                bool remove_noise)
{
  Ort::Value &output = output_tensors.at(0); // (1,3,h,w)
  const unsigned int h = mat.rows;
  const unsigned int w = mat.cols;

  auto output_dims = output.GetTypeInfo().GetTensorTypeAndShapeInfo().GetShape();
  const unsigned int out_h = output_dims.at(2);
  const unsigned int out_w = output_dims.at(3);
  const unsigned int channel_step = out_h * out_w;

  float *output_ptr = output.GetTensorMutableData<float>();

  std::vector<float> elements(channel_step, 0.f); // allocate
  for (unsigned int i = 0; i < channel_step; ++i)
    elements[i] = (float) argmax(output_ptr + i, channel_step); // with normalize

  cv::Mat mask(out_h, out_w, CV_32FC1, elements.data()); // ref only !
  // post process
  if (remove_noise) lite::utils::remove_small_connected_area(mask, 0.05f);
  // already allocated a new continuous memory after resize.
  if (out_h != h || out_w != w) cv::resize(mask, mask, cv::Size(w, h));
    // need clone to allocate a new continuous memory if not performed resize.
    // The memory elements point to will release after return.
  else mask = mask.clone();

  content.mask = mask; // auto handle the memory inside ocv with smart ref.
  content.flag = true;
}