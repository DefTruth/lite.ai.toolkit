//
// Created by DefTruth on 2022/5/15.
//

#ifndef LITE_AI_TOOLKIT_MNN_CV_MNN_IRIS_LANDMARKS_H
#define LITE_AI_TOOLKIT_MNN_CV_MNN_IRIS_LANDMARKS_H

#include "lite/mnn/core/mnn_core.h"

namespace mnncv
{
  class LITE_EXPORTS MNNIrisLandmarks
  {
  public:
    explicit MNNIrisLandmarks(const std::string &_mnn_path, unsigned int _num_threads = 1);

    ~MNNIrisLandmarks() noexcept;

  private:
    std::shared_ptr<MNN::Interpreter> mnn_interpreter;
    MNN::Session *mnn_session = nullptr;
    MNN::Tensor *input_tensor = nullptr; // assume single input.
    MNN::ScheduleConfig schedule_config;
    const char *mnn_path = nullptr;
    const char *log_id = nullptr;
    const unsigned int num_threads; // initialize at runtime.

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
    // hardcode input size
    const int input_batch = 1;
    const int input_channel = 3;
    const int input_height = 64;
    const int input_width = 64;

  private:
    void transform(cv::Mat &mat_rs); // without resize

    void print_debug_string();

    void resize_unscale(const cv::Mat &mat,
                        cv::Mat &mat_rs,
                        int target_height,
                        int target_width,
                        IrisScaleParams &scale_params);

    void generate_eye_contours_brows_and_iris_landmarks3d(
        const IrisScaleParams &scale_params,
        const std::map<std::string, MNN::Tensor *> &output_tensors,
        types::Landmarks3D &eyes_contours_and_brows,
        types::Landmarks3D &iris, int img_height,
        int img_width);

  public:
    void detect(const cv::Mat &mat, types::Landmarks3D &eyes_contours_and_brows,
                types::Landmarks3D &iris, bool is_screen_right_eye = false);
  };
}

#endif //LITE_AI_TOOLKIT_MNN_CV_MNN_IRIS_LANDMARKS_H
