//
// Created by DefTruth on 2022/6/19.
//

#ifndef LITE_AI_TOOLKIT_ORT_CV_PORTRAIT_SEG_EXTREMEC3NET_H
#define LITE_AI_TOOLKIT_ORT_CV_PORTRAIT_SEG_EXTREMEC3NET_H

#include "lite/ort/core/ort_core.h"

namespace ortcv
{
  class LITE_EXPORTS PortraitSegExtremeC3Net : public BasicOrtHandler
  {
  public:
    explicit PortraitSegExtremeC3Net(const std::string &_onnx_path, unsigned int _num_threads = 1) :
        BasicOrtHandler(_onnx_path, _num_threads)
    {};

    ~PortraitSegExtremeC3Net() override = default;

  private:
    // nested classes
    typedef struct
    {
      float r;
      int dw;
      int dh;
      int new_unpad_w;
      int new_unpad_h;
      bool flag;
    } PortraitSegExtremeC3NetScaleParams;

  private:
    const float mean_vals[3] = {107.304565f, 115.69884f, 132.35703f}; // BGR
    const float scale_vals[3] = {1.f / (63.97182f * 255.f), 1.f / (65.1337f * 255.f),
                                 1.f / (68.29726f * 255.f)};

  private:
    Ort::Value transform(const cv::Mat &mat_rs) override;

    void resize_unscale(const cv::Mat &mat, cv::Mat &mat_rs,
                        int target_height, int target_width,
                        PortraitSegExtremeC3NetScaleParams &scale_params);

    void generate_mask(const PortraitSegExtremeC3NetScaleParams &scale_params,
                       std::vector<Ort::Value> &output_tensors,
                       const cv::Mat &mat, types::PortraitSegContent &content,
                       float score_threshold = 0.0f, bool remove_noise = false);

  public:
    void detect(const cv::Mat &mat, types::PortraitSegContent &content,
                float score_threshold = 0.0f, bool remove_noise = false);
  };
}


#endif //LITE_AI_TOOLKIT_ORT_CV_PORTRAIT_SEG_EXTREMEC3NET_H
