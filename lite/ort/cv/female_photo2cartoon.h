//
// Created by DefTruth on 2022/6/3.
//

#ifndef LITE_AI_TOOLKIT_ORT_CV_FEMALE_PHOTO2CARTOON_H
#define LITE_AI_TOOLKIT_ORT_CV_FEMALE_PHOTO2CARTOON_H

#include "lite/ort/core/ort_core.h"

namespace ortcv
{
  // Static single input and single output.
  class LITE_EXPORTS FemalePhoto2Cartoon : public BasicOrtHandler
  {
  public:
    explicit FemalePhoto2Cartoon(const std::string &_onnx_path, unsigned int _num_threads = 1) :
        BasicOrtHandler(_onnx_path, _num_threads)
    {};

    ~FemalePhoto2Cartoon() override = default;

  private:
    Ort::Value transform(const cv::Mat &mat_merged_rs /*merged & resized mat*/) override;

    void generate_cartoon(std::vector<Ort::Value> &output_tensors,
                          const cv::Mat &mask_rs,
                          types::FemalePhoto2CartoonContent &content);

  public:
    void detect(const cv::Mat &mat, const cv::Mat &mask, types::FemalePhoto2CartoonContent &content);
  };
}

#endif //LITE_AI_TOOLKIT_ORT_CV_FEMALE_PHOTO2CARTOON_H
