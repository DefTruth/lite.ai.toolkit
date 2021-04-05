//
// Created by DefTruth on 2021/4/3.
//

#ifndef LITEHUB_ORT_CV_EMOTION_FERPLUS_H
#define LITEHUB_ORT_CV_EMOTION_FERPLUS_H

#include "ort/core/ort_core.h"

namespace ortcv
{
  class EmotionFerPlus : public BasicOrtHandler
  {
  private:
    const char *emotion_texts[8] = {
        "netural", "happiness", "surprise", "sadness", "anger",
        "disgust", "fear", "contempt"
    };
  public:
    explicit EmotionFerPlus(const std::string &_onnx_path, unsigned int _num_threads = 1) :
        BasicOrtHandler(_onnx_path, _num_threads)
    {};

    ~EmotionFerPlus()
    {};

  private:
    ort::Value transform(const cv::Mat &mat);

  public:
    void detect(const cv::Mat &mat, types::Emotions &emotions);
  };
}
#endif //LITEHUB_ORT_CV_EMOTION_FERPLUS_H
