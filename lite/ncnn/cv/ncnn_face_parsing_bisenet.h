//
// Created by DefTruth on 2022/7/2.
//

#ifndef LITE_AI_TOOLKIT_NCNN_CV_NCNN_FACE_PARSING_BISENET_H
#define LITE_AI_TOOLKIT_NCNN_CV_NCNN_FACE_PARSING_BISENET_H

#include "lite/ncnn/core/ncnn_core.h"

namespace ncnncv
{
  class LITE_EXPORTS NCNNFaceParsingBiSeNet : public BasicNCNNHandler
  {
  public:
    explicit NCNNFaceParsingBiSeNet(const std::string &_param_path,
                                    const std::string &_bin_path,
                                    unsigned int _num_threads = 1,
                                    unsigned int _input_height = 512,
                                    unsigned int _input_width = 512);

    ~NCNNFaceParsingBiSeNet() override = default;

  private:
    const int input_height;
    const int input_width;
    const float mean_vals[3] = {0.485f * 255.f, 0.456f * 255.f, 0.406f * 255.f}; // RGB
    const float norm_vals[3] = {1.f / (0.229f * 255.f), 1.f / (0.224f * 255.f), 1.f / (0.225f * 255.f)};

  private:
    void transform(const cv::Mat &mat, ncnn::Mat &in) override;

    void generate_mask(ncnn::Extractor &extractor,
                       const cv::Mat &mat, types::FaceParsingContent &content,
                       bool minimum_post_process = false);

  public:
    void detect(const cv::Mat &mat, types::FaceParsingContent &content,
                bool minimum_post_process = false);

  };
}

#endif //LITE_AI_TOOLKIT_NCNN_CV_NCNN_FACE_PARSING_BISENET_H
