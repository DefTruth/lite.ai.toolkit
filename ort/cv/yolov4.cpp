//
// Created by DefTruth on 2021/3/14.
//

#include "yolov4.h"
#include "ort/core/ort_utils.h"

using ortcv::YoloV4;


ort::Value YoloV4::transform(const cv::Mat &mat)
{
  cv::Mat canva = mat.clone();
  cv::cvtColor(canva, canva, cv::COLOR_BGR2RGB);
  cv::resize(canva, canva, cv::Size(input_node_dims.at(3),
                                    input_node_dims.at(2)));
  // (1,3,640|416,640|416) 1xCXHXW

  ortcv::utils::transform::normalize_inplace(canva, mean_val, scale_val); // float32
  return ortcv::utils::transform::create_tensor(
      canva, input_node_dims, memory_info_handler,
      input_values_handler, ortcv::utils::transform::CHW);
}