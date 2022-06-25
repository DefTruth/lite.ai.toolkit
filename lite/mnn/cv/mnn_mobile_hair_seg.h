//
// Created by DefTruth on 2022/6/22.
//

#ifndef LITE_AI_TOOLKIT_MNN_CV_MNN_MOBILE_HAIR_SEG_H
#define LITE_AI_TOOLKIT_MNN_CV_MNN_MOBILE_HAIR_SEG_H

#include "lite/mnn/core/mnn_core.h"

namespace mnncv
{
  class LITE_EXPORTS MNNMobileHairSeg : public BasicMNNHandler
  {
  public:
    explicit MNNMobileHairSeg(const std::string &_mnn_path, unsigned int _num_threads = 1); //
    ~MNNMobileHairSeg() override = default;

  private:
    const float mean_vals[3] = {127.5f, 127.5f, 127.5f};
    const float norm_vals[3] = {1.f / 127.5f, 1.f / 127.5f, 1.f / 127.5f};

  private:
    void initialize_pretreat();

    void transform(const cv::Mat &mat) override; // resize & normalize.

    void generate_mask(const std::map<std::string, MNN::Tensor *> &output_tensors,
                       const cv::Mat &mat, types::HairSegContent &content,
                       float score_threshold = 0.0f, bool remove_noise = false);

  public:
    void detect(const cv::Mat &mat, types::HairSegContent &content,
                float score_threshold = 0.0f, bool remove_noise = false);
  };
}

#endif //LITE_AI_TOOLKIT_MNN_CV_MNN_MOBILE_HAIR_SEG_H
