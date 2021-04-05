//
// Created by DefTruth on 2021/4/5.
//

#ifndef LITEHUB_ORT_CV_SUBPIXEL_CNN_H
#define LITEHUB_ORT_CV_SUBPIXEL_CNN_H

#include "ort/core/ort_core.h"

namespace ortcv
{
  class SubPixelCNN : public BasicOrtHandler
  {
  public:
    explicit SubPixelCNN(const std::string &_onnx_path, unsigned int _num_threads = 1) :
        BasicOrtHandler(_onnx_path, _num_threads)
    {};

    ~SubPixelCNN()
    {};

  private:
    ort::Value transform(const cv::Mat &mat);

  public:
    void detect(const cv::Mat &mat, types::SuperResolutionContent &super_resolution_content);
  };
}

#endif //LITEHUB_ORT_CV_SUBPIXEL_CNN_H
