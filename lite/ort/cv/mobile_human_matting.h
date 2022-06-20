//
// Created by DefTruth on 2022/6/20.
//

#ifndef LITE_AI_TOOLKIT_ORT_CV_MOBILE_HUMAN_MATTING_H
#define LITE_AI_TOOLKIT_ORT_CV_MOBILE_HUMAN_MATTING_H

#include "lite/ort/core/ort_core.h"

namespace ortcv
{
  class LITE_EXPORTS MobileHumanMatting : public BasicOrtHandler
  {
  public:
    explicit MobileHumanMatting(const std::string &_onnx_path, unsigned int _num_threads = 1) :
        BasicOrtHandler(_onnx_path, _num_threads)
    {};

    ~MobileHumanMatting() override = default;

  private:
    const float mean_vals[3] = {104.f, 112.f, 121.f}; // BGR
    const float scale_vals[3] = {1.f / 255.f, 1.f / 255.f, 1.f / 255.f};

  private:
    Ort::Value transform(const cv::Mat &mat) override;

    void generate_matting(std::vector<Ort::Value> &output_tensors,
                          const cv::Mat &mat, types::MattingContent &content,
                          bool remove_noise = false, bool minimum_post_process = false);

  public:
    void detect(const cv::Mat &mat, types::MattingContent &content, bool remove_noise = false,
                bool minimum_post_process = false);

  };
}

#endif //LITE_AI_TOOLKIT_ORT_CV_MOBILE_HUMAN_MATTING_H
