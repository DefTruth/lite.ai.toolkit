//
// Created by DefTruth on 2021/3/14.
//

#ifndef LITEHUB_ORT_CV_FSANET_H
#define LITEHUB_ORT_CV_FSANET_H

#include "ort/core/ort_core.h"

namespace ortcv
{

  class FSANet : public BasicOrtHandler
  {

  private:
    static constexpr const float pad = 0.3f;
    static constexpr const int input_width = 64;
    static constexpr const int input_height = 64;

  public:
    explicit FSANet(const std::string &_onnx_path, unsigned int _num_threads = 1) :
        BasicOrtHandler(_onnx_path, _num_threads)
    {};

    ~FSANet()
    {}; // override

  private:
    ort::Value transform(const cv::Mat &mat); //  padding & resize & normalize.

  public:
    void detect(const cv::Mat &mat, types::EulerAngles &euler_angles);
  };

}

#endif //LITEHUB_ORT_CV_FSANET_H
