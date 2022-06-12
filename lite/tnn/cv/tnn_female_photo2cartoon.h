//
// Created by DefTruth on 2022/6/12.
//

#ifndef LITE_AI_TOOLKIT_TNN_CV_TNN_FEMALE_PHOTO2CARTOON_H
#define LITE_AI_TOOLKIT_TNN_CV_TNN_FEMALE_PHOTO2CARTOON_H

#include "lite/tnn/core/tnn_core.h"

namespace tnncv
{
  class LITE_EXPORTS TNNFemalePhoto2Cartoon : public BasicTNNHandler
  {
  public:
    explicit TNNFemalePhoto2Cartoon(const std::string &_proto_path,
                                    const std::string &_model_path,
                                    unsigned int _num_threads = 1);

    ~TNNFemalePhoto2Cartoon() override = default;

  private:
    // In TNN: x*scale + bias
    std::vector<float> scale_vals = {1.f / 127.5f, 1.f / 127.5f, 1.f / 127.5f};
    std::vector<float> bias_vals = {-1.f, -1.f, -1.f};

  private:
    void transform(const cv::Mat &mat_merged_rs /*merged & resized mat*/) override;

    void generate_cartoon(std::shared_ptr<tnn::Instance> &_instance,
                          const cv::Mat &mask_rs, types::FemalePhoto2CartoonContent &content);

  public:
    void detect(const cv::Mat &mat, const cv::Mat &mask, types::FemalePhoto2CartoonContent &content);
  };
}

#endif //LITE_AI_TOOLKIT_TNN_CV_TNN_FEMALE_PHOTO2CARTOON_H
