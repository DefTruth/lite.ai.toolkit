//
// Created by YanJun Qiu on 2021/3/28.
//

#include "ort_utils.h"

//*************************************** ortcv::utils **********************************************//
// reference: https://github.com/DefTruth/headpose-fsanet-pytorch/blob/master/src/utils.py
void ortcv::utils::draw_axis_inplace(cv::Mat &mat_inplace,
                                     const types::EulerAngles &euler_angles,
                                     float size, int thickness) {
  const float pitch = euler_angles.pitch * _PI / 180.f;
  const float yaw = -euler_angles.yaw * _PI / 180.f;
  const float roll = euler_angles.roll * _PI / 180.f;

  const float height = static_cast<float>(mat_inplace.rows);
  const float width = static_cast<float>(mat_inplace.cols);

  const int tdx = static_cast<int>(width / 2.0f);
  const int tdy = static_cast<int>(height / 2.0f);

  // X-Axis pointing to right. drawn in red
  const int x1 = static_cast<int>(size * std::cosf(yaw) * std::cosf(roll)) + tdx;
  const int y1 = static_cast<int>(
                     size * (std::cosf(pitch) * std::sinf(roll)
                             + std::cosf(roll) * std::sinf(pitch) * std::sinf(yaw))
                 ) + tdy;
  // Y-Axis | drawn in green
  const int x2 = static_cast<int>(-size * std::cosf(yaw) * std::sinf(roll)) + tdx;
  const int y2 = static_cast<int>(
                     size * (std::cosf(pitch) * std::cosf(roll)
                             - std::sinf(pitch) * std::sinf(yaw) * std::sinf(roll))
                 ) + tdy;
  // Z-Axis (out of the screen) drawn in blue
  const int x3 = static_cast<int>(size * std::sinf(yaw)) + tdx;
  const int y3 = static_cast<int>(-size * std::cosf(yaw) * std::sinf(pitch)) + tdy;

  cv::line(mat_inplace, cv::Point2i(tdx, tdy), cv::Point2i(x1, y1), cv::Scalar(0, 0, 255), thickness);
  cv::line(mat_inplace, cv::Point2i(tdx, tdy), cv::Point2i(x2, y2), cv::Scalar(0, 255, 0), thickness);
  cv::line(mat_inplace, cv::Point2i(tdx, tdy), cv::Point2i(x3, y3), cv::Scalar(255, 0, 0), thickness);
}

cv::Mat ortcv::utils::draw_axis(const cv::Mat &mat,
                                const types::EulerAngles &euler_angles,
                                float size, int thickness) {
  cv::Mat mat_copy = mat.clone();

  const float pitch = euler_angles.pitch * _PI / 180.f;
  const float yaw = -euler_angles.yaw * _PI / 180.f;
  const float roll = euler_angles.roll * _PI / 180.f;

  const float height = static_cast<float>(mat_copy.rows);
  const float width = static_cast<float>(mat_copy.cols);

  const int tdx = static_cast<int>(width / 2.0f);
  const int tdy = static_cast<int>(height / 2.0f);

  // X-Axis pointing to right. drawn in red
  const int x1 = static_cast<int>(size * std::cosf(yaw) * std::cosf(roll)) + tdx;
  const int y1 = static_cast<int>(
                     size * (std::cosf(pitch) * std::sinf(roll)
                             + std::cosf(roll) * std::sinf(pitch) * std::sinf(yaw))
                 ) + tdy;
  // Y-Axis | drawn in green
  const int x2 = static_cast<int>(-size * std::cosf(yaw) * std::sinf(roll)) + tdx;
  const int y2 = static_cast<int>(
                     size * (std::cosf(pitch) * std::cosf(roll)
                             - std::sinf(pitch) * std::sinf(yaw) * std::sinf(roll))
                 ) + tdy;
  // Z-Axis (out of the screen) drawn in blue
  const int x3 = static_cast<int>(size * std::sinf(yaw)) + tdx;
  const int y3 = static_cast<int>(-size * std::cosf(yaw) * std::sinf(pitch)) + tdy;

  cv::line(mat_copy, cv::Point2i(tdx, tdy), cv::Point2i(x1, y1), cv::Scalar(0, 0, 255), thickness);
  cv::line(mat_copy, cv::Point2i(tdx, tdy), cv::Point2i(x2, y2), cv::Scalar(0, 255, 0), thickness);
  cv::line(mat_copy, cv::Point2i(tdx, tdy), cv::Point2i(x3, y3), cv::Scalar(255, 0, 0), thickness);

  return mat_copy;
}

cv::Mat ortcv::utils::draw_landmarks(const cv::Mat &mat, types::Landmarks &landmarks) {
  if (landmarks.empty()) return mat;
  cv::Mat mat_copy = mat.clone();
  for (const auto &point: landmarks)
    cv::circle(mat_copy, point, 2, cv::Scalar(0, 255, 0), -1);
  return mat_copy;
}

void ortcv::utils::draw_landmarks_inplace(cv::Mat &mat, types::Landmarks &landmarks) {
  if (landmarks.empty()) return;
  for (const auto &point: landmarks)
    cv::circle(mat, point, 2, cv::Scalar(0, 255, 0), -1);
}

void ortcv::utils::draw_boxes_inplace(cv::Mat &mat_inplace, const std::vector<types::Boxf> &boxes) {
  if (boxes.empty()) return;
  for (const auto &box: boxes) {
    cv::rectangle(mat_inplace, box.rect(), cv::Scalar(255, 255, 0), 2);
  }
}

cv::Mat ortcv::utils::draw_boxes(const cv::Mat &mat, const std::vector<types::Boxf> &boxes) {
  if (boxes.empty()) return mat;
  cv::Mat canva = mat.clone();
  for (const auto &box: boxes) {
    cv::rectangle(canva, box.rect(), cv::Scalar(255, 255, 0), 2);
  }
  return canva;
}

// reference: https://github.com/Linzaer/Ultra-Light-Fast-Generic-Face-Detector-1MB/
//            blob/master/ncnn/src/UltraFace.cpp
void ortcv::utils::hard_nms(std::vector<types::Boxf> &input, std::vector<types::Boxf> &output,
                            float iou_threshold, unsigned int topk) {
  if (input.empty()) return;
  std::sort(input.begin(), input.end(),
            [](const types::Boxf &a, const types::Boxf &b) { return a.score > b.score; });
  const unsigned int box_num = input.size();
  std::vector<int> merged(box_num, 0);

  unsigned int count = 0;
  for (unsigned int i = 0; i < box_num; ++i) {
    if (merged[i]) continue;
    std::vector<types::Boxf> buf;

    buf.push_back(input[i]);
    merged[i] = 1;

    for (unsigned int j = i + 1; j < box_num; ++j) {
      if (merged[j]) continue;

      float iou = static_cast<float>(input[i].iou_of(input[j]));

      if (iou > iou_threshold) {
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

void ortcv::utils::blending_nms(std::vector<types::Boxf> &input, std::vector<types::Boxf> &output,
                                float iou_threshold, unsigned int topk) {
  if (input.empty()) return;
  std::sort(input.begin(), input.end(),
            [](const types::Boxf &a, const types::Boxf &b) { return a.score > b.score; });
  const unsigned int box_num = input.size();
  std::vector<int> merged(box_num, 0);

  unsigned int count = 0;
  for (unsigned int i = 0; i < box_num; ++i) {
    if (merged[i]) continue;
    std::vector<types::Boxf> buf;

    buf.push_back(input[i]);
    merged[i] = 1;

    for (unsigned int j = i + 1; j < box_num; ++j) {
      if (merged[j]) continue;

      float iou = static_cast<float>(input[i].iou_of(input[j]));
      if (iou > iou_threshold) {
        merged[j] = 1;
        buf.push_back(input[j]);
      }
    }

    float total = 0.f;
    for (unsigned int k = 0; k < buf.size(); ++k) {
      total += std::expf(buf[k].score);
    }
    types::Boxf rects;
    for (unsigned int l = 0; l < buf.size(); ++l) {
      float rate = std::expf(buf[l].score) / total;
      rects.x1 += buf[l].x1 * rate;
      rects.y1 += buf[l].y1 * rate;
      rects.x2 += buf[l].x2 * rate;
      rects.y2 += buf[l].y2 * rate;
      rects.score += buf[l].score * rate;
    }
    output.push_back(rects);

    // keep top k
    count += 1;
    if (count >= topk)
      break;
  }
}

ort::Value ortcv::utils::transform::mat3f_to_tensor(const cv::Mat &mat3f,
                                                    const std::vector<int64_t> &tensor_dims,
                                                    const ort::MemoryInfo &memory_info_handler,
                                                    std::vector<float> &tensor_value_handler,
                                                    unsigned int data_format)
throw(std::runtime_error) {
  cv::Mat mat3f_ref;
  if (mat3f.type() != CV_32FC3) mat3f.convertTo(mat3f_ref, CV_32FC3);
  else mat3f_ref = mat3f; // reference only. zero-time cost.

  const unsigned int rows = mat3f_ref.rows;
  const unsigned int cols = mat3f_ref.cols;
  const unsigned int channels = mat3f_ref.channels();

  if (tensor_dims.size() != 4) throw std::runtime_error("dims mismatch.");
  if (tensor_dims.at(0) != 1) throw std::runtime_error("batch != 1");

  // CXHXW
  if (data_format == transform::CHW) {

    const unsigned int target_channel = tensor_dims.at(1);
    const unsigned int target_height = tensor_dims.at(2);
    const unsigned int target_width = tensor_dims.at(3);
    const unsigned int target_tensor_size = target_channel * target_height * target_width;
    if (target_channel != channels) throw std::runtime_error("channel mismatch.");

    tensor_value_handler.resize(target_tensor_size);

    cv::Mat resize_mat_ref;
    if (target_height != rows || target_width != cols)
      cv::resize(mat3f_ref, resize_mat_ref, cv::Size(target_width, target_height));
    else resize_mat_ref = mat3f_ref; // reference only. zero-time cost.

    std::vector<cv::Mat> mat_channels;
    cv::split(resize_mat_ref, mat_channels);
    std::vector<float> channel_values;
    channel_values.resize(target_height * target_width);
    for (unsigned int i = 0; i < channels; ++i) {
      channel_values.clear();
      channel_values = mat_channels.at(i).reshape(1, 1); // flatten
      std::memcpy(tensor_value_handler.data() + i * (target_height * target_width),
                  channel_values.data(),
                  target_height * target_width * sizeof(float)); // CXHXW
    }

    return ort::Value::CreateTensor<float>(memory_info_handler, tensor_value_handler.data(),
                                           target_tensor_size, tensor_dims.data(),
                                           tensor_dims.size());
  }

  // HXWXC
  const unsigned int target_channel = tensor_dims.at(3);
  const unsigned int target_height = tensor_dims.at(1);
  const unsigned int target_width = tensor_dims.at(2);
  const unsigned int target_tensor_size = target_channel * target_height * target_width;
  if (target_channel != channels) throw std::runtime_error("channel mismatch!");
  tensor_value_handler.clear();

  cv::Mat resize_mat_ref;
  if (target_height != rows || target_width != cols)
    cv::resize(mat3f_ref, resize_mat_ref, cv::Size(target_width, target_height));
  else resize_mat_ref = mat3f_ref; // reference only. zero-time cost.

  tensor_value_handler.assign(resize_mat_ref.data, resize_mat_ref.data + target_tensor_size);

  return ort::Value::CreateTensor<float>(memory_info_handler, tensor_value_handler.data(),
                                         target_tensor_size, tensor_dims.data(),
                                         tensor_dims.size());
}

cv::Mat ortcv::utils::transform::normalize(const cv::Mat &mat, float mean, float scale) {
  cv::Mat matf;
  if (mat.type() != CV_32FC3) mat.convertTo(matf, CV_32FC3);
  else matf = mat; // reference
  return (matf - mean) * scale;
}

cv::Mat ortcv::utils::transform::normalize(const cv::Mat &mat, float *mean, float *scale) {
  cv::Mat mat_copy;
  if (mat.type() != CV_32FC3) mat.convertTo(mat_copy, CV_32FC3);
  else mat_copy = mat.clone();
  for (unsigned int i = 0; i < mat_copy.rows; ++i) {
    cv::Vec3f *p = mat_copy.ptr<cv::Vec3f>(i);
    for (unsigned int j = 0; j < mat_copy.cols; ++i) {
      p[j][0] = (p[j][0] - mean[0]) * scale[0];
      p[j][1] = (p[j][1] - mean[1]) * scale[1];
      p[j][2] = (p[j][2] - mean[2]) * scale[2];
    }
  }
  return mat_copy;
}

void ortcv::utils::transform::normalize(const cv::Mat &inmat, cv::Mat &outmat,
                                        float mean, float scale) {
  outmat = ortcv::utils::transform::normalize(inmat, mean, scale);
}

void ortcv::utils::transform::normalize_inplace(cv::Mat &mat_inplace, float mean, float scale) {
  if (mat_inplace.type() != CV_32FC3) mat_inplace.convertTo(mat_inplace, CV_32FC3);
  ortcv::utils::transform::normalize(mat_inplace, mat_inplace, mean, scale);
}

void ortcv::utils::transform::normalize_inplace(cv::Mat &mat_inplace, float *mean, float *scale) {
  if (mat_inplace.type() != CV_32FC3) mat_inplace.convertTo(mat_inplace, CV_32FC3);
  for (unsigned int i = 0; i < mat_inplace.rows; ++i) {
    cv::Vec3f *p = mat_inplace.ptr<cv::Vec3f>(i);
    for (unsigned int j = 0; j < mat_inplace.cols; ++i) {
      p[j][0] = (p[j][0] - mean[0]) * scale[0];
      p[j][1] = (p[j][1] - mean[1]) * scale[1];
      p[j][2] = (p[j][2] - mean[2]) * scale[2];
    }
  }
}


//*************************************** ortcv::utils **********************************************//









