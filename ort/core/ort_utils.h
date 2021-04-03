//
// Created by DefTruth on 2021/3/28.
//

#ifndef LITEHUB_ORT_CORE_ORT_UTILS_H
#define LITEHUB_ORT_CORE_ORT_UTILS_H

#include "__ort_core.h"
#include "ort_types.h"

namespace ortcv {
  namespace utils {
    static constexpr const float _PI = 3.1415926f;
    cv::Mat draw_axis(const cv::Mat &mat, const types::EulerAngles &euler_angles, float size = 50.f, int thickness = 2);
    cv::Mat draw_boxes(const cv::Mat &mat, const std::vector<types::Boxf> &boxes);
    cv::Mat draw_landmarks(const cv::Mat &mat, types::Landmarks &landmarks);
    cv::Mat draw_age(const cv::Mat &mat, types::Age &age);
    void draw_boxes_inplace(cv::Mat &mat_inplace, const std::vector<types::Boxf> &boxes);
    void draw_axis_inplace(cv::Mat &mat_inplace, const types::EulerAngles &euler_angles,float size = 50.f, int thickness = 2);
    void draw_landmarks_inplace(cv::Mat &mat, types::Landmarks &landmarks);
    void draw_age_inplace(cv::Mat &mat_inplace, types::Age &age);
    void hard_nms(std::vector<types::Boxf> &input, std::vector<types::Boxf> &output,float iou_threshold, unsigned int topk);
    void blending_nms(std::vector<types::Boxf> &input, std::vector<types::Boxf> &output, float iou_threshold, unsigned int topk);
    namespace transform {
      enum {CHW = 0, HWC = 1};
      /**
       * @param mat3f CV:Mat with type 'CV_32FC3'
       * @param tensor_dims e.g {1,C,H,W} | {1,H,W,C}
       * @param memory_info It needs to be a global variable in a class
       * @param tensor_value_handler It needs to be a global variable in a class
       * @param data_format CHW | HWC
       * @return
       */
      ort::Value mat3f_to_tensor(const cv::Mat &mat3f, const std::vector<int64_t> &tensor_dims,
                                 const ort::MemoryInfo &memory_info_handler,
                                 std::vector<float> &tensor_value_handler,
                                 unsigned int data_format = CHW) throw(std::runtime_error);
      cv::Mat normalize(const cv::Mat &mat, float mean, float scale);
      void normalize(const cv::Mat &inmat, cv::Mat &outmat, float mean, float scale);
      cv::Mat normalize(const cv::Mat &mat, const float mean[3], const float scale[3]);
      void normalize_inplace(cv::Mat &mat_inplace, float mean, float scale);
      void normalize_inplace(cv::Mat &mat_inplace, const float mean[3], const float scale[3]);
    }

    namespace math {
      template<typename T=float> std::vector<T> softmax(const std::vector<T> &logits, unsigned int &max_id);
      template<typename T=float> std::vector<T> softmax(const T *logits, unsigned int _size, unsigned int &max_id);
    }

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

template<typename T>
std::vector<T> ortcv::utils::math::softmax(const T *logits, unsigned int _size, unsigned int &max_id) {
  ::__assert_type<T>();
  if (_size == 0) return {};
  T max_prob = static_cast<T>(0), total_exp = static_cast<T>(0);
  std::vector<float> softmax_probs(_size);
  for (unsigned int i = 0; i < _size; ++i) {
    softmax_probs[i] = std::expf(logits[i]);
    total_exp += softmax_probs[i];
  }
  for (unsigned int i = 0; i < _size; ++i) {
    softmax_probs[i] = softmax_probs[i] / total_exp;
    if (softmax_probs[i] > max_prob) {
      max_id = i;
      max_prob = softmax_probs[i];
    }
  }
  return softmax_probs;
}

template<typename T>
std::vector<T> ortcv::utils::math::softmax(const std::vector<T> &logits, unsigned int &max_id) {
  ::__assert_type<T>();
  if (logits.empty()) return {};
  const unsigned int _size = logits.size();
  T max_prob = static_cast<T>(0), total_exp = static_cast<T>(0);
  std::vector<float> softmax_probs(_size);
  for (unsigned int i = 0; i < _size; ++i) {
    softmax_probs[i] = std::expf(logits[i]);
    total_exp += softmax_probs[i];
  }
  for (unsigned int i = 0; i < _size; ++i) {
    softmax_probs[i] = softmax_probs[i] / total_exp;
    if (softmax_probs[i] > max_prob) {
      max_id = i;
      max_prob = softmax_probs[i];
    }
  }
  return softmax_probs;
}

#endif //LITEHUB_ORT_CORE_ORT_UTILS_H
