//
// Created by DefTruth on 2022/5/2.
//

#ifndef LITE_AI_TOOLKIT_ORT_CV_IRIS_LANDMARKS_H
#define LITE_AI_TOOLKIT_ORT_CV_IRIS_LANDMARKS_H

#include "lite/ort/core/ort_core.h"

namespace ortcv
{
  class LITE_EXPORTS IrisLandmarks : public BasicOrtHandler
  {
  public:
    explicit IrisLandmarks(const std::string &_onnx_path, unsigned int _num_threads = 1) :
        BasicOrtHandler(_onnx_path, _num_threads)
    {};

    ~IrisLandmarks() override = default; // override

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
    } IrisScaleParams;

  private:
    static constexpr const float mean_val = 0.f;
    static constexpr const float scale_val = 1.0f / 255.0f; // RGB

  private:
    Ort::Value transform(const cv::Mat &mat_rs) override;

    void resize_unscale(const cv::Mat &mat,
                        cv::Mat &mat_rs,
                        int target_height,
                        int target_width,
                        IrisScaleParams &scale_params);

    void generate_eye_contours_brows_and_iris_landmarks3d(
        const IrisScaleParams &scale_params,
        std::vector<Ort::Value> &output_tensors,
        types::Landmarks3D &eyes_contours_and_brows,
        types::Landmarks3D &iris, int img_height,
        int img_width);

  public:
    void detect(const cv::Mat &mat, types::Landmarks3D &eyes_contours_and_brows,
                types::Landmarks3D &iris, bool is_screen_right_eye = false);
  };
}


#endif //LITE_AI_TOOLKIT_ORT_CV_IRIS_LANDMARKS_H
