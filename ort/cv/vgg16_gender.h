//
// Created by DefTruth on 2021/4/4.
//

#ifndef LITEHUB_ORT_CV_VGG16_GENDER_H
#define LITEHUB_ORT_CV_VGG16_GENDER_H

#include "ort/core/ort_core.h"

namespace ortcv {
  class VGG16Gender: public BasicOrtHandler {
  private:
    const char * gender_texts[2] = {"female", "male"};

  public:
    VGG16Gender(const std::string &_onnx_path, unsigned int _num_threads = 1):
        BasicOrtHandler(_onnx_path, _num_threads) {};
    ~VGG16Gender() {};

  private:
    ort::Value transform(const cv::Mat &mat);

  public:
    void detect(const cv::Mat &mat, types::Gender &gender);
  };
}
#endif //LITEHUB_ORT_CV_VGG16_GENDER_H
