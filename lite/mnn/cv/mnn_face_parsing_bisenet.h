//
// Created by DefTruth on 2022/6/30.
//

#ifndef LITE_AI_TOOLKIT_MNN_CV_MNN_FACE_PARSING_BISENET_H
#define LITE_AI_TOOLKIT_MNN_CV_MNN_FACE_PARSING_BISENET_H

#include "lite/mnn/core/mnn_core.h"

namespace mnncv
{
  class LITE_EXPORTS MNNFaceParsingBiSeNet : public BasicMNNHandler
  {
  public:
    explicit MNNFaceParsingBiSeNet(const std::string &_mnn_path, unsigned int _num_threads = 1); //
    ~MNNFaceParsingBiSeNet() override = default;

  private:
    const float mean_vals[3] = {0.485f * 255.f, 0.456f * 255.f, 0.406f * 255.f}; // RGB
    const float norm_vals[3] = {1.f / (0.229f * 255.f), 1.f / (0.224f * 255.f), 1.f / (0.225f * 255.f)};

  private:
    void initialize_pretreat();

    void transform(const cv::Mat &mat) override; // resize & normalize.

    void generate_mask(const std::map<std::string, MNN::Tensor *> &output_tensors,
                       const cv::Mat &mat, types::FaceParsingContent &content,
                       bool minimum_post_process = false);

  public:
    void detect(const cv::Mat &mat, types::FaceParsingContent &content,
                bool minimum_post_process = false);
  };
}

#endif //LITE_AI_TOOLKIT_MNN_CV_MNN_FACE_PARSING_BISENET_H
