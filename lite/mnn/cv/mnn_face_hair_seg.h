//
// Created by DefTruth on 2022/6/19.
//

#ifndef LITE_AI_TOOLKIT_MNN_CV_MNN_FACE_HAIR_SEG_H
#define LITE_AI_TOOLKIT_MNN_CV_MNN_FACE_HAIR_SEG_H

#include "lite/mnn/core/mnn_core.h"

namespace mnncv
{
  class LITE_EXPORTS MNNFaceHairSeg : public BasicMNNHandler
  {
  public:
    explicit MNNFaceHairSeg(const std::string &_mnn_path, unsigned int _num_threads = 1); //
    ~MNNFaceHairSeg() override = default;

  private:
    const float mean_vals[3] = {0.f, 0.f, 0.f};
    const float norm_vals[3] = {1.f / 255.f, 1.f / 255.f, 1.f / 255.f};

  private:
    void initialize_pretreat();

    void transform(const cv::Mat &mat) override; // resize & normalize.

    void generate_mask(const std::map<std::string, MNN::Tensor *> &output_tensors,
                       const cv::Mat &mat, types::FaceHairSegContent &content,
                       bool remove_noise = false);

  public:
    void detect(const cv::Mat &mat, types::FaceHairSegContent &content,
                bool remove_noise = false);
  };
}

#endif //LITE_AI_TOOLKIT_MNN_CV_MNN_FACE_HAIR_SEG_H
