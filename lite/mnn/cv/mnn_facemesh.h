//
// Created by DefTruth on 2022/5/15.
//

#ifndef LITE_AI_TOOLKIT_MNN_CV_MNN_FACEMESH_H
#define LITE_AI_TOOLKIT_MNN_CV_MNN_FACEMESH_H

#include "lite/mnn/core/mnn_core.h"

namespace mnncv
{
  class LITE_EXPORTS MNNFaceMesh
  {
    explicit MNNFaceMesh(const std::string &_mnn_path, unsigned int _num_threads = 1);

    ~MNNFaceMesh() noexcept;

  private:
    std::shared_ptr<MNN::Interpreter> mnn_interpreter;
    MNN::Session *mnn_session = nullptr;
    MNN::Tensor *input_tensor = nullptr; // assume single input.
    MNN::ScheduleConfig schedule_config;
    const char *mnn_path = nullptr;
    const char *log_id = nullptr;
    const unsigned int num_threads; // initialize at runtime.
    int dimension_type; // hint only

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
    // hardcode input size
    const int input_batch = 1;
    const int input_channel = 3;
    const int input_height = 192;
    const int input_width = 192;

  private:
    void transform(cv::Mat &mat_rs); // without resize

    void print_debug_string();

    void resize_unscale(const cv::Mat &mat,
                        cv::Mat &mat_rs,
                        int target_height,
                        int target_width,
                        FaceMeshScaleParams &scale_params);

    void generate_landmarks3d_and_confidence(
        const FaceMeshScaleParams &scale_params,
        const std::map<std::string, MNN::Tensor *> &output_tensors,
        types::Landmarks3D &landmarks3d, float &confidence,
        int img_height, int img_width);

  public:
    void detect(const cv::Mat &mat, types::Landmarks3D &landmarks3d, float &confidence);
  };
}

#endif //LITE_AI_TOOLKIT_MNN_CV_MNN_FACEMESH_H
