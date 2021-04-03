//
// Created by DefTruth on 2021/4/3.
//

#ifndef LITEHUB_ORT_CV_GENDER_GOOGLENET_H
#define LITEHUB_ORT_CV_GENDER_GOOGLENET_H

#include "ort/core/ort_core.h"

namespace ortcv {
  class GenderGoogleNet: public BasicOrtHandler {
  private:
    const float mean_val[3] = {104.0f, 117.0f, 123.0f};
    const float scale_val[3] = {1.0f, 1.0f, 1.0f};
    const char * gender_texts[2] = {"female", "male"};
  public:
    GenderGoogleNet(const std::string &_onnx_path, unsigned int _num_threads = 1) :
        BasicOrtHandler(_onnx_path, _num_threads) {};

    ~GenderGoogleNet() {};

  private:
    ort::Value transform(const cv::Mat &mat);

  public:
    void detect(const cv::Mat &mat, types::Gender &gender);
  };
}
#endif //LITEHUB_ORT_CV_GENDER_GOOGLENET_H
