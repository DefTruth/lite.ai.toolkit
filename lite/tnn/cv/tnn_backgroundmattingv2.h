//
// Created by DefTruth on 2022/4/9.
//

#ifndef LITE_AI_TOOLKIT_TNN_CV_TNN_BACKGROUNDMATTINGV2_H
#define LITE_AI_TOOLKIT_TNN_CV_TNN_BACKGROUNDMATTINGV2_H

#include "lite/tnn/core/tnn_core.h"

namespace tnncv
{
  class LITE_EXPORTS TNNBackgroundMattingV2
  {
  public:
    explicit TNNBackgroundMattingV2(const std::string &_proto_path,
                                    const std::string &_model_path,
                                    unsigned int _num_threads = 1);

    ~TNNBackgroundMattingV2();

  private:
    const char *log_id = nullptr;
    const char *proto_path = nullptr;
    const char *model_path = nullptr;
    // Note, tnn:: actually is TNN_NS::, I prefer the first one.
    std::shared_ptr<tnn::TNN> net;
    std::shared_ptr<tnn::Instance> instance;

  private:
    std::vector<float> scale_vals = {1.f / 255.f, 1.f / 255.f, 1.f / 255.f};
    std::vector<float> bias_vals = {0.f, 0.f, 0.f}; // RGB
    // hardcode input node names, hint only.
    // downsample_ratio has been freeze while onnx exported
    // and, the input size of each input has been freeze, also.
    std::vector<const char *> input_names = {
        "src",
        "bgr"
    };
    // hardcode output node names, hint only.
    std::vector<const char *> output_names = {
        "pha",
        "fgr",
        "pha_sm",
        "fgr_sm",
        "err_sm",
        "ref_sm"
    };

  private:
    const unsigned int num_threads; // initialize at runtime.
    // multi inputs, rxi will be update inner video matting process.
    std::shared_ptr<tnn::Mat> src_mat;
    std::shared_ptr<tnn::Mat> bgr_mat;
    int input_height;
    int input_width;
    tnn::DataFormat input_data_format;  // e.g DATA_FORMAT_NHWC
    tnn::MatType input_mat_type; // e.g NCHW_FLOAT
    tnn::DeviceType input_device_type; // only CPU, namely ARM or X86
    tnn::DeviceType output_device_type; // only CPU, namely ARM or X86
    tnn::DeviceType network_device_type; // e.g DEVICE_X86 DEVICE_NAIVE DEVICE_ARM
    std::map<std::string, tnn::DimsVector> input_shapes;
    std::map<std::string, tnn::DimsVector> output_shapes;

    // un-copyable
  protected:
    TNNBackgroundMattingV2(const TNNBackgroundMattingV2 &) = delete; //
    TNNBackgroundMattingV2(TNNBackgroundMattingV2 &&) = delete; //
    TNNBackgroundMattingV2 &operator=(const TNNBackgroundMattingV2 &) = delete; //
    TNNBackgroundMattingV2 &operator=(TNNBackgroundMattingV2 &&) = delete; //

  private:
    void print_debug_string(); // debug information

  private:
    void transform(const cv::Mat &mat_rs, const cv::Mat &bgr_rs);

    void initialize_instance(); // init net & instance

    void generate_matting(std::shared_ptr<tnn::Instance> &_instance,
                          const cv::Mat &mat, types::MattingContent &content,
                          bool remove_noise = false, bool minimum_post_process = false);

  public:
    /**
     * @param mat cv::Mat input image with BGR format.
     * @param bgr cv::Mat input background image with BGR format.
     * @param content MattingContent output fgr, pha and merge_mat (if minimum_post_process is false)
     * @param remove_noise bool, whether to remove small connected areas.
     * @param minimum_post_process bool, will not return demo merge mat if True.
     */
    void detect(const cv::Mat &mat, const cv::Mat &bgr, types::MattingContent &content,
                bool remove_noise = false, bool minimum_post_process = false);


  };
}

#endif //LITE_AI_TOOLKIT_TNN_CV_TNN_BACKGROUNDMATTINGV2_H
