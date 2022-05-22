//
// Created by DefTruth on 2022/5/2.
//

#ifndef LITE_AI_TOOLKIT_ORT_CV_FACEMESH_H
#define LITE_AI_TOOLKIT_ORT_CV_FACEMESH_H

#include "lite/ort/core/ort_core.h"

namespace ortcv
{
  class LITE_EXPORTS FaceMesh : public BasicOrtHandler
  {
  public:
    explicit FaceMesh(const std::string &_onnx_path, unsigned int _num_threads = 1) :
        BasicOrtHandler(_onnx_path, _num_threads)
    {};

    ~FaceMesh() override = default; // override

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
    } FaceMeshScaleParams;

  private:
    static constexpr const float mean_val = 0.f;
    static constexpr const float scale_val = 1.0f / 255.0f; // RGB

  private:
    Ort::Value transform(const cv::Mat &mat_rs) override;

    void resize_unscale(const cv::Mat &mat,
                        cv::Mat &mat_rs,
                        int target_height,
                        int target_width,
                        FaceMeshScaleParams &scale_params);

    void generate_landmarks3d_and_confidence(
        const FaceMeshScaleParams &scale_params,
        std::vector<Ort::Value> &output_tensors,
        types::Landmarks3D &landmarks3d,
        float &confidence, int img_height,
        int img_width);

  public:
    void detect(const cv::Mat &mat, types::Landmarks3D &landmarks3d, float &confidence);
  };
}

#endif //LITE_AI_TOOLKIT_ORT_CV_FACEMESH_H
