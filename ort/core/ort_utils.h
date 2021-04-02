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

    cv::Mat draw_axis(const cv::Mat &mat, const types::EulerAngles &euler_angles,
                      float size = 50.f, int thickness = 2);

    cv::Mat draw_boxes(const cv::Mat &mat, const std::vector<types::Boxf> &boxes);

    cv::Mat draw_landmarks(const cv::Mat &mat, types::Landmarks &landmarks);

    cv::Mat draw_age(const cv::Mat &mat, types::Age &age);

    void draw_boxes_inplace(cv::Mat &mat_inplace, const std::vector<types::Boxf> &boxes);

    void draw_axis_inplace(cv::Mat &mat_inplace, const types::EulerAngles &euler_angles,
                           float size = 50.f, int thickness = 2);

    void draw_landmarks_inplace(cv::Mat &mat, types::Landmarks &landmarks);

    void draw_age_inplace(cv::Mat &mat_inplace, types::Age &age);

    void hard_nms(std::vector<types::Boxf> &input, std::vector<types::Boxf> &output,
                  float iou_threshold, unsigned int topk);

    void blending_nms(std::vector<types::Boxf> &input, std::vector<types::Boxf> &output,
                      float iou_threshold, unsigned int topk);

    namespace transform {
      enum {
        CHW = 0, HWC = 1
      };

      /**
       *
       * @param mat3f CV:Mat with type 'CV_32FC3'
       * @param tensor_dims e.g {1,C,H,W} | {1,H,W,C}
       * @param memory_info It needs to be a global variable in a class
       * @param tensor_value_handler It needs to be a global variable in a class
       * @param data_format CHW | HWC
       * @return
       */
      ort::Value mat3f_to_tensor(const cv::Mat &mat3f,
                                 const std::vector<int64_t> &tensor_dims,
                                 const ort::MemoryInfo &memory_info_handler,
                                 std::vector<float> &tensor_value_handler,
                                 unsigned int data_format = CHW)
      throw(std::runtime_error);

      cv::Mat normalize(const cv::Mat &mat, float mean, float scale);

      void normalize(const cv::Mat &inmat, cv::Mat &outmat, float mean, float scale);

      cv::Mat normalize(const cv::Mat &mat, const float mean[3], const float scale[3]);

      void normalize_inplace(cv::Mat &mat_inplace, float mean, float scale);

      void normalize_inplace(cv::Mat &mat_inplace, const float mean[3], const float scale[3]);

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

#endif //LITEHUB_ORT_CORE_ORT_UTILS_H
