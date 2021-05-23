//
// Created by DefTruth on 2021/5/23.
//

#include "yolov3.h"
#include "ort/core/ort_utils.h"

using ortcv::YoloV3;

std::vector<ort::Value> YoloV3::transform(const std::vector<cv::Mat> &mats)
{
  cv::Mat canvas = mats.at(0).clone();  // (h,w,3) uint8 mats contains one mat only.
  cv::cvtColor(canvas, canvas, cv::COLOR_BGR2RGB);
  // multi inputs: input_1 image_shape
  // input_1 with shape (1,3,416,416);
  // image_shape is original shape of source image.
  std::vector<int64_t> input_1_dims = input_node_dims.at(0); // (1,3,416,416); reference
  std::vector<int64_t> image_shape_dims = input_node_dims.at(1); // (1,2);
  const unsigned int input_height = input_1_dims.at(2); // 416
  const unsigned int input_width = input_1_dims.at(3); // 416
  const unsigned int image_height = canvas.rows;
  const unsigned int image_width = canvas.cols;

  const float scale = std::fmin(
      (float) input_width / (float) image_width,
      (float) input_height / (float) image_height
  );

  const unsigned int nw = static_cast<unsigned int>((float) image_width * scale);
  const unsigned int nh = static_cast<unsigned int>((float) image_height * scale);

  cv::resize(canvas, canvas, cv::Size(nw, nh));

  cv::Mat canvas_pad(input_height, input_width, CV_8UC3, 128);
  const unsigned int x1 = (input_width - nw) / 2;
  const unsigned int y1 = (input_height - nh) / 2;
  cv::Rect roi(x1, y1, nw, nh);
  canvas.convertTo(canvas_pad(roi), CV_8UC3); // padding

  std::vector<ort::Value> input_tensors;
  // make tensor of input_1 & image_shape
  ortcv::utils::transform::normalize_inplace(canvas_pad, mean_val, scale_val); // float32 (0.,1.)
  input_tensors.emplace_back(ortcv::utils::transform::create_tensor(
      canvas_pad, input_1_dims, memory_info_handler,
      input_values_handlers.at(0), ortcv::utils::transform::CHW
  )); // input_1
  auto image_shape_values = input_values_handlers.at(1); // reference
  image_shape_values[0] = static_cast<float>(image_height);
  image_shape_values[1] = static_cast<float>(image_width);
  input_tensors.emplace_back(ort::Value::CreateTensor<float>(
      memory_info_handler, input_values_handlers.at(1).data(),
      input_tensor_sizes.at(1), image_shape_dims.data(),
      image_shape_dims.size())
  ); // image_shape

  return input_tensors;
}