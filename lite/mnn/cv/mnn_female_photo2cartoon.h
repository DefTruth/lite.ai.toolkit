//
// Created by DefTruth on 2022/6/12.
//

#ifndef LITE_AI_TOOLKIT_MNN_CV_MNN_FEMALE_PHOTO2CARTOON_H
#define LITE_AI_TOOLKIT_MNN_CV_MNN_FEMALE_PHOTO2CARTOON_H

#include "lite/mnn/core/mnn_core.h"

namespace mnncv
{
  class LITE_EXPORTS MNNFemalePhoto2Cartoon : public BasicMNNHandler
  {
  public:
    explicit MNNFemalePhoto2Cartoon(const std::string &_mnn_path, unsigned int _num_threads = 1); //
    ~MNNFemalePhoto2Cartoon() override = default;

  private:
    const float mean_vals[3] = {127.5f, 127.5f, 127.5f};
    const float norm_vals[3] = {1.f / 127.5f, 1.f / 127.5f, 1.f / 127.5f};

  private:
    void initialize_pretreat();

    void transform(const cv::Mat &mat_merged_rs /*merged & resized mat*/) override;

    void generate_cartoon(const std::map<std::string, MNN::Tensor *> &output_tensors,
                          const cv::Mat &mask_rs, types::FemalePhoto2CartoonContent &content);

  public:
    void detect(const cv::Mat &mat, const cv::Mat &mask, types::FemalePhoto2CartoonContent &content);
  };
}

#endif //LITE_AI_TOOLKIT_MNN_CV_MNN_FEMALE_PHOTO2CARTOON_H
