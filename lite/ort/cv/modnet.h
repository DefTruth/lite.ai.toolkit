//
// Created by DefTruth on 2022/3/27.
//

#ifndef LITE_AI_TOOLKIT_ORT_CV_MODNET_H
#define LITE_AI_TOOLKIT_ORT_CV_MODNET_H

#include "lite/ort/core/ort_core.h"

namespace ortcv
{
  class LITE_EXPORTS MODNet : public BasicOrtHandler
  {
  public:
    explicit MODNet(const std::string &_onnx_path, unsigned int _num_threads = 1) :
        BasicOrtHandler(_onnx_path, _num_threads)
    {};

    ~MODNet() override = default;

  private:
    static constexpr const float mean = 127.5f;
    static constexpr const float scale = 1.f / 127.5f;

  private:
    Ort::Value transform(const cv::Mat &mat) override;

  public:
    void detect(const cv::Mat &mat, types::MattingContent &content);
  };
}

#endif //LITE_AI_TOOLKIT_ORT_CV_MODNET_H
