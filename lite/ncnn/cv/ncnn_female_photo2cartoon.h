//
// Created by DefTruth on 2022/6/12.
//

#ifndef LITE_AI_TOOLKIT_NCNN_CV_NCNN_FEMALE_PHOTO2CARTOON_H
#define LITE_AI_TOOLKIT_NCNN_CV_NCNN_FEMALE_PHOTO2CARTOON_H

#include "lite/ncnn/core/ncnn_core.h"

namespace ncnncv
{
  class LITE_EXPORTS NCNNFemalePhoto2Cartoon : public BasicNCNNHandler
  {
  public:
    explicit NCNNFemalePhoto2Cartoon(const std::string &_param_path,
                                     const std::string &_bin_path,
                                     unsigned int _num_threads = 1,
                                     unsigned int _input_height = 256,
                                     unsigned int _input_width = 256);

    ~NCNNFemalePhoto2Cartoon() override = default;

  private:
    const int input_height;
    const int input_width;
    const float mean_vals[3] = {127.5f, 127.5f, 127.5f};
    const float norm_vals[3] = {1.f / 127.5f, 1.f / 127.5f, 1.f / 127.5f};

  private:
    void transform(const cv::Mat &mat_merged_rs /*merged & resized mat*/, ncnn::Mat &in) override;

    void generate_cartoon(ncnn::Extractor &extractor, const cv::Mat &mask_rs,
                          types::FemalePhoto2CartoonContent &content);

  public:
    void detect(const cv::Mat &mat, const cv::Mat &mask, types::FemalePhoto2CartoonContent &content);
  };
}


#endif //LITE_AI_TOOLKIT_NCNN_CV_NCNN_FEMALE_PHOTO2CARTOON_H
