//
// Created by DefTruth on 2022/7/2.
//

#ifndef LITE_AI_TOOLKIT_TNN_CV_TNN_FACE_PARSING_BISENET_H
#define LITE_AI_TOOLKIT_TNN_CV_TNN_FACE_PARSING_BISENET_H

#include "lite/tnn/core/tnn_core.h"

namespace tnncv
{
  class LITE_EXPORTS TNNFaceParsingBiSeNet : public BasicTNNHandler
  {
  public:
    explicit TNNFaceParsingBiSeNet(const std::string &_proto_path,
                                   const std::string &_model_path,
                                   unsigned int _num_threads = 1);

    ~TNNFaceParsingBiSeNet() override = default;

  private:
    // In TNN: x*scale + bias
    std::vector<float> bias_vals = {-0.485f * 255.f, -0.456f * 255.f, -0.406f * 255.f}; // RGB
    std::vector<float> scale_vals = {1.f / (0.229f * 255.f), 1.f / (0.224f * 255.f), 1.f / (0.225f * 255.f)};

  private:
    void transform(const cv::Mat &mat_rs) override; //

    void generate_mask(std::shared_ptr<tnn::Instance> &_instance,
                       const cv::Mat &mat, types::FaceParsingContent &content,
                       bool minimum_post_process = false);

  public:
    void detect(const cv::Mat &mat, types::FaceParsingContent &content,
                bool minimum_post_process = false);

  };
}

#endif //LITE_AI_TOOLKIT_TNN_CV_TNN_FACE_PARSING_BISENET_H
