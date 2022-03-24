//
// Created by DefTruth on 2022/3/20.
//

#ifndef LITE_AI_TOOLKIT_ORT_CV_PIPNET19_H
#define LITE_AI_TOOLKIT_ORT_CV_PIPNET19_H

#include "lite/ort/core/ort_core.h"

namespace ortcv
{
  class LITE_EXPORTS PIPNet19 : public BasicOrtHandler
  {
  public:
    explicit PIPNet19(const std::string &_onnx_path, unsigned int _num_threads = 1) :
        BasicOrtHandler(_onnx_path, _num_threads)
    {};

    ~PIPNet19() override = default;

  private:
    // x=(x-mean)*std
    const float mean_vals[3] = {0.485f, 0.456f, 0.406f}; // RGB order
    const float scale_vals[3] = {1.0 / 0.229f, 1.0 / 0.224f, 1.0 / 0.225f};
    static constexpr const unsigned int num_nb = 10;
    static constexpr const unsigned int num_lms = 19;
    static constexpr const unsigned int max_len = 18;
    static constexpr const unsigned int net_stride = 32;

  private:
    Ort::Value transform(const cv::Mat &mat) override;

    void generate_landmarks(types::Landmarks &landmarks,
                            std::vector<Ort::Value> &output_tensors,
                            float img_height, float img_width);

  public:
    void detect(const cv::Mat &mat, types::Landmarks &landmarks);

  private:
    const unsigned int reverse_index1[19 * 18] = {
        1, 2, 6, 7, 8, 1, 2, 6, 7, 8, 1, 2, 6, 7, 8, 1, 2, 6, 0, 2, 3, 4, 6, 7, 8, 0, 2, 3, 4, 6, 7, 8, 0, 2, 3, 4, 0, 1, 3, 4, 5, 6, 7, 8,
        9, 10, 11, 12, 0, 1, 3, 4, 5, 6, 0, 1, 2, 4, 5, 6, 7, 8, 9, 10, 11, 14, 0, 1, 2, 4, 5, 6, 1, 2, 3, 5, 9, 10, 11, 1, 2, 3, 5, 9, 10,
        11, 1, 2, 3, 5, 3, 4, 9, 10, 11, 3, 4, 9, 10, 11, 3, 4, 9, 10, 11, 3, 4, 9, 0, 1, 2, 3, 7, 8, 12, 13, 15, 0, 1, 2, 3, 7, 8, 12, 13,
        15, 0, 1, 2, 3, 4, 5, 6, 8, 9, 10, 11, 12, 13, 15, 16, 18, 0, 1, 0, 1, 2, 3, 4, 5, 6, 7, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 0,
        1, 2, 3, 4, 5, 6, 7, 8, 10, 11, 12, 13, 14, 15, 16, 17, 18, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 11, 13, 14, 16, 17, 18, 0, 1, 3, 4, 5, 9,
        10, 14, 17, 3, 4, 5, 9, 10, 14, 17, 3, 4, 5, 9, 0, 1, 6, 7, 8, 13, 14, 15, 16, 17, 18, 0, 1, 6, 7, 8, 13, 14, 0, 2, 5, 6, 7, 8, 9,
        10, 11, 12, 14, 15, 16, 17, 18, 0, 2, 5, 4, 5, 9, 10, 11, 12, 13, 15, 16, 17, 18, 4, 5, 9, 10, 11, 12, 13, 12, 13, 14, 16, 17, 18,
        12, 13, 14, 16, 17, 18, 12, 13, 14, 16, 17, 18, 12, 13, 14, 15, 17, 18, 12, 13, 14, 15, 17, 18, 12, 13, 14, 15, 17, 18, 12, 13, 14,
        15, 16, 18, 12, 13, 14, 15, 16, 18, 12, 13, 14, 15, 16, 18, 15, 16, 17, 15, 16, 17, 15, 16, 17, 15, 16, 17, 15, 16, 17, 15, 16, 17
    };
    const unsigned int reverse_index2[19 * 18] = {
        0, 6, 1, 4, 6, 0, 6, 1, 4, 6, 0, 6, 1, 4, 6, 0, 6, 1, 0, 1, 8, 7, 2, 2, 3, 0, 1, 8, 7, 2, 2, 3, 0, 1, 8, 7, 3, 1, 3, 5, 5, 4, 3, 1,
        5, 6, 6, 9, 3, 1, 3, 5, 5, 4, 5, 5, 3, 1, 3, 7, 5, 5, 1, 3, 4, 9, 5, 5, 3, 1, 3, 7, 7, 8, 1, 0, 3, 2, 2, 7, 8, 1, 0, 3, 2, 2, 7, 8,
        1, 0, 6, 0, 6, 4, 1, 6, 0, 6, 4, 1, 6, 0, 6, 4, 1, 6, 0, 6, 1, 3, 4, 9, 1, 2, 6, 9, 8, 1, 3, 4, 9, 1, 2, 6, 9, 8, 2, 2, 2, 7, 8, 9,
        0, 0, 9, 9, 9, 5, 7, 7, 8, 8, 2, 2, 4, 4, 0, 5, 6, 6, 3, 0, 4, 5, 7, 4, 3, 8, 6, 6, 9, 6, 7, 6, 5, 0, 4, 4, 8, 6, 4, 0, 3, 8, 4, 4,
        9, 7, 6, 7, 9, 8, 7, 2, 2, 2, 9, 9, 9, 0, 0, 8, 5, 9, 7, 9, 9, 8, 4, 3, 1, 2, 1, 6, 8, 4, 3, 1, 2, 1, 6, 8, 4, 3, 1, 2, 6, 9, 5, 7,
        8, 0, 2, 1, 3, 4, 4, 6, 9, 5, 7, 8, 0, 2, 8, 9, 8, 6, 8, 7, 7, 8, 8, 0, 0, 2, 2, 2, 5, 8, 9, 8, 9, 7, 8, 7, 5, 2, 1, 4, 4, 1, 3, 9,
        7, 8, 7, 5, 2, 1, 1, 5, 7, 0, 3, 1, 1, 5, 7, 0, 3, 1, 1, 5, 7, 0, 3, 1, 3, 2, 3, 0, 0, 0, 3, 2, 3, 0, 0, 0, 3, 2, 3, 0, 0, 0, 7, 6,
        1, 3, 1, 2, 7, 6, 1, 3, 1, 2, 7, 6, 1, 3, 1, 2, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5
    };

  };
}
#endif //LITE_AI_TOOLKIT_ORT_CV_PIPNET19_H
