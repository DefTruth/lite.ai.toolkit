//
// Created by DefTruth on 2022/6/11.
//

#ifndef LITE_AI_TOOLKIT_TNN_CV_TNN_HEAD_SEG_H
#define LITE_AI_TOOLKIT_TNN_CV_TNN_HEAD_SEG_H

#include "lite/tnn/core/tnn_core.h"

namespace tnncv
{
  class LITE_EXPORTS TNNHeadSeg
  {
  public:
    explicit TNNHeadSeg(const std::string &_proto_path,
                        const std::string &_model_path,
                        unsigned int _num_threads = 1);

    ~TNNHeadSeg();

  private:
    const char *log_id = nullptr;
    const char *proto_path = nullptr;
    const char *model_path = nullptr;
    // Note, tnn:: actually is TNN_NS::, I prefer the first one.
    std::shared_ptr<tnn::TNN> net;
    std::shared_ptr<tnn::Instance> instance;
    std::shared_ptr<tnn::Mat> input_mat;
    const unsigned int num_threads; // initialize at runtime.

  private:
    // y = scale*x + bias
    std::vector<float> scale_vals = {1.f / 255.f, 1.f / 255.f, 1.f / 255.f};
    std::vector<float> bias_vals = {0.f, 0.f, 0.f}; // RGB

  private:
    // input size (1,384,384,3)
    unsigned int input_batch = 1;
    unsigned int input_channel = 3;
    unsigned int input_height = 384;
    unsigned int input_width = 384;

  private:
    tnn::DataFormat input_data_format;  // e.g DATA_FORMAT_NHWC
    tnn::MatType input_mat_type; // e.g NCHW_FLOAT
    tnn::DeviceType input_device_type; // only CPU, namely ARM or X86
    tnn::DeviceType output_device_type; // only CPU, namely ARM or X86
    tnn::DeviceType network_device_type; // e.g DEVICE_X86 DEVICE_NAIVE DEVICE_ARM
    tnn::DimsVector input_shape; // debug
    tnn::DimsVector output_shape;

    // un-copyable
  protected:
    TNNHeadSeg(const TNNHeadSeg &) = delete; //
    TNNHeadSeg(TNNHeadSeg &&) = delete; //
    TNNHeadSeg &operator=(const TNNHeadSeg &) = delete; //
    TNNHeadSeg &operator=(TNNHeadSeg &&) = delete; //

  private:
    void print_debug_string(); // debug information

  private:
    void transform(const cv::Mat &mat_rs); //

    void initialize_instance(); // init net & instance

  public:
    void detect(const cv::Mat &mat, types::HeadSegContent &content);
  };
}

#endif //LITE_AI_TOOLKIT_TNN_CV_TNN_HEAD_SEG_H
