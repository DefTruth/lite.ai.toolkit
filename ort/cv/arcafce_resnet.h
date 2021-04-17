//
// Created by DefTruth on 2021/4/4.
//

#ifndef LITEHUB_ORT_CV_ARCAFCE_RESNET_H
#define LITEHUB_ORT_CV_ARCAFCE_RESNET_H

#include "ort/core/ort_core.h"

namespace ortcv
{
  class ArcFaceResNet : public BasicOrtHandler
  {
  public:
    explicit ArcFaceResNet(const std::string &_onnx_path, unsigned int _num_threads = 1) :
        BasicOrtHandler(_onnx_path, _num_threads)
    {};

    ~ArcFaceResNet() override = default;

  private:
    ort::Value transform(const cv::Mat &mat) override;

  public:
    void detect(const cv::Mat &mat, types::FaceContent &face_content);
  };
}

#endif //LITEHUB_ORT_CV_ARCAFCE_RESNET_H
