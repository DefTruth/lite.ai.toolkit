//
// Created by DefTruth on 2022/4/9.
//

#ifndef LITE_AI_TOOLKIT_MNN_CV_MNN_BACKGROUNDMATTINGV2_H
#define LITE_AI_TOOLKIT_MNN_CV_MNN_BACKGROUNDMATTINGV2_H

#include "lite/mnn/core/mnn_core.h"

namespace mnncv
{
  class LITE_EXPORTS MNNBackgroundMattingV2
  {
  public:
    explicit MNNBackgroundMattingV2(const std::string &_mnn_path,
                                    unsigned int _num_threads = 1); //
    ~MNNBackgroundMattingV2();

  private:
    std::shared_ptr<MNN::Interpreter> mnn_interpreter;
    MNN::Session *mnn_session = nullptr;
    MNN::ScheduleConfig schedule_config;
    std::shared_ptr<MNN::CV::ImageProcess> pretreat; // init at runtime
    const char *log_id = nullptr;
    const char *mnn_path = nullptr;

  private:
    const float mean_vals[3] = {0.f, 0.f, 0.f}; // RGB
    const float norm_vals[3] = {1.f / 255.f, 1.f / 255.f, 1.f / 255.f};
    // hardcode input node names, hint only.
    // downsample_ratio has been freeze while onnx exported
    // and, the input size of each input has been freeze, also.
    std::vector<const char *> input_node_names = {
        "src",
        "bgr"
    };
    // hardcode output node names, hint only.
    std::vector<const char *> output_node_names = {
        "pha",
        "fgr",
        "pha_sm",
        "fgr_sm",
        "err_sm",
        "ref_sm"
    };

  private:
    const unsigned int num_threads; // initialize at runtime.
    // multi inputs.
    MNN::Tensor *src_tensor = nullptr;
    MNN::Tensor *bgr_tensor = nullptr;
    // input size, initialize at runtime.
    int input_height;
    int input_width;
    int dimension_type; // hint only

    // un-copyable
  protected:
    MNNBackgroundMattingV2(const MNNBackgroundMattingV2 &) = delete; //
    MNNBackgroundMattingV2(MNNBackgroundMattingV2 &&) = delete; //
    MNNBackgroundMattingV2 &operator=(const MNNBackgroundMattingV2 &) = delete; //
    MNNBackgroundMattingV2 &operator=(MNNBackgroundMattingV2 &&) = delete; //

  private:
    void print_debug_string();

  private:
    void transform(const cv::Mat &mat, const cv::Mat &bgr);

    void initialize_pretreat(); //

    void initialize_interpreter();

    void generate_matting(const std::map<std::string, MNN::Tensor *> &output_tensors,
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
#endif //LITE_AI_TOOLKIT_MNN_CV_MNN_BACKGROUNDMATTINGV2_H
