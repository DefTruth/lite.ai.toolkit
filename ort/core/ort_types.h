//
// Created by YanJun Qiu on 2021/3/28.
//

#ifndef LITEHUB_CORE_ORT_TYPES_H
#define LITEHUB_CORE_ORT_TYPES_H

#include <limits>
#include <cmath>
#include <vector>
#include <type_traits>
#include "opencv2/opencv.hpp"

namespace ortcv {
  namespace types {
    // 1. bounding box.
    template<typename T1 = float, typename T2 = float>
    struct BoundingBox {
      T1 x1;
      T1 y1;
      T1 x2;
      T1 y2;
      T2 score;
      // convert type.
      template<typename O1, typename O2 = T2>
      BoundingBox<O1, O2> convert() const;
      template<typename O1, typename O2 = T2>
      T1 iou_of(const BoundingBox<O1, O2> &other) const;
      T1 width() const;
      T1 height() const;
      T1 area() const;
      cv::Rect rect() const;
    }; // End BoundingBox.
    // specific alias.
    typedef BoundingBox<int> Boxi;
    typedef BoundingBox<float> Boxf;
    typedef BoundingBox<double> Boxd;
    // 2. euler angles.
    typedef struct { float yaw; float pitch; float roll;} EulerAngles;
  } // NAMESPACE TYPES
} // NAMESPACE ORTCV

namespace ortnlp {
  namespace types {} // NAMESPACE TYPES
} // NAMESPACE ORTNLP

namespace ortasr {
  namespace types {} // NAMESPACE TYPES
} // NAMESPACE ORTASR

/**
 * Implemeantation details for all custom types using on liteort.
 */

/* implementation for 'BoundingBox'. */
template<typename TT1, typename TT2 = float>
static void assert_support_type() {
  static_assert(std::is_pod<TT1>::value && std::is_pod<TT2>::value
                && std::is_floating_point<TT2>::value, "not support type.");
} // only support for some specific types. check at complie-time.

template<typename T1, typename T2>
template<typename O1, typename O2>
ortcv::types::BoundingBox<O1, O2> ortcv::types::BoundingBox<T1, T2>::convert() const {
  assert_support_type<T1, T2>();
  assert_support_type<O1, O2>();
  BoundingBox<O1, O2> other;
  other.x1 = static_cast<O1>(x1);
  other.y1 = static_cast<O1>(y1);
  other.x2 = static_cast<O1>(x2);
  other.y2 = static_cast<O1>(y2);
  other.score = static_cast<O2>(score);
  return other;
}

template<typename T1, typename T2>
template<typename O1, typename O2>
T1 ortcv::types::BoundingBox<T1, T2>::iou_of(const BoundingBox<O1, O2> &other) const {
  BoundingBox<T1, T2> tbox = other.template convert<T1, T2>();
  T1 inner_x1 = x1 > tbox.x1 ? x1: tbox.x1;
  T1 inner_y1 = y1 > tbox.y1 ? y1: tbox.y1;
  T1 inner_x2 = x2 < tbox.x2 ? x2: tbox.x2;
  T1 inner_y2 = y2 < tbox.y2 ? y2: tbox.y2;
  T1 inner_h = inner_y2 - inner_y1 + static_cast<T1>(1.0f);
  T1 inner_w = inner_x2 - inner_x1 + static_cast<T1>(1.0f);
  if (inner_h <= 0.f || inner_w <= 0.f)
    return std::numeric_limits<T1>::min();
  T1 inner_area = inner_h * inner_w;
  T1 w1 = tbox.x2 - x1 + static_cast<T1>(1.0f);
  T1 h1 = tbox.y2 - y1 + static_cast<T1>(1.0f);
  T1 area1 = h1 * w1;
  return inner_area / (area() + area1 - inner_area);
}

template<typename T1, typename T2>
cv::Rect ortcv::types::BoundingBox<T1, T2>::rect() const {
  assert_support_type<T1, T2>();
  BoundingBox<int> boxi = this->template convert<int>();
  return cv::Rect(boxi.x1, boxi.y1, boxi.width(), boxi.height());
}

template<typename T1, typename T2>
T1 ortcv::types::BoundingBox<T1, T2>::width() const
{assert_support_type<T1, T2>(); return (x2 - x1 + static_cast<T1>(1)); }

template<typename T1, typename T2>
T1 ortcv::types::BoundingBox<T1, T2>::height() const
{assert_support_type<T1, T2>(); return (y2 - y1 + static_cast<T1>(1)); }

template<typename T1, typename T2>
T1 ortcv::types::BoundingBox<T1, T2>::area() const
{return std::abs<T1>(width() * height());}

#endif //LITEHUB_CORE_ORT_TYPES_H
