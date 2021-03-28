//
// Created by YanJun Qiu on 2021/3/28.
//

#ifndef LITEHUB_CORE_ORT_UTILS_H
#define LITEHUB_CORE_ORT_UTILS_H

#include "ort_types.h"

namespace ortcv {
  namespace utils {
    static constexpr const float _PI = 3.1415926f;
    cv::Mat draw_axis(const cv::Mat &mat, const types::EulerAngles &euler_angles,
                      float size = 50.f, int thickness = 2);
    cv::Mat draw_boxes(const cv::Mat &mat, const std::vector<types::Boxf> &boxes);
    void draw_boxes_inplace(cv::Mat &mat_inplace, const std::vector<types::Boxf> &boxes);
    void draw_axis_inplace(cv::Mat &mat_inplace, const types::EulerAngles &euler_angles,
                           float size = 50.f, int thickness = 2);
    void hard_nms(std::vector<types::Boxf> &input, std::vector<types::Boxf> &output,
                  float iou_threshold, int topk);
    void blending_nms(std::vector<types::Boxf> &input, std::vector<types::Boxf> &output,
                      float iou_threshold, int topk);

  } // NAMESPACE UTILS
} // NAMESPACE ORTCV

namespace ortnlp {
  namespace utils {

  } // NAMESPACE UTILS
} // NAMESPACE ORTNLP

namespace ortasr {
  namespace utils {

  } // NAMESPACE UTILS
} // NAMESPACE ORTASR

#endif //LITEHUB_CORE_ORT_UTILS_H
