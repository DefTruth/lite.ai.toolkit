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
void ortcv::utils::hard_nms(std::vector<types::Boxf> &input,
                            std::vector<types::Boxf> &output,
                            float iou_threshold, int topk) {
  if (input.empty()) return;
  std::sort(input.begin(), input.end(),
            [](const types::Boxf &a, const types::Boxf &b) { return a.score > b.score; });
  const unsigned int box_num = input.size();
  std::vector<int> merged(box_num, 0);

  int count = 0;
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

void ortcv::utils::blending_nms(std::vector<types::Boxf> &input,
                                std::vector<types::Boxf> &output,
                                float iou_threshold, int topk) {
  if (input.empty()) return;
  std::sort(input.begin(), input.end(),
            [](const types::Boxf &a, const types::Boxf &b) { return a.score > b.score; });
  const unsigned int box_num = input.size();
  std::vector<int> merged(box_num, 0);

  int count = 0;
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
//*************************************** ortcv::utils **********************************************//









