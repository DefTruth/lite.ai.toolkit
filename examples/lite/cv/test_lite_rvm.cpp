//
// Created by DefTruth on 2021/9/20.
//

#include "lite/lite.h"

static void test_default()
{
  std::string onnx_path = "../../../hub/onnx/cv/rvm_mobilenetv3_fp32.onnx";
  std::string video_path = "../../../examples/lite/resources/test_lite_rvm_0.mp4";
  std::string background_path = "../../../examples/lite/resources/test_lite_matting_bgr.jpg";
  std::string output_path = "../../../logs/test_lite_rvm_0.mp4";

  cv::Mat background = cv::imread(background_path);
  auto *rvm = new lite::cv::matting::RobustVideoMatting(onnx_path, 16); // 16 threads
  std::vector<lite::types::MattingContent> contents;

  // 1. video matting.
  rvm->detect_video(video_path, output_path, contents, false, 0.4f,
                    20, true, true, background);

  delete rvm;
}

static void test_onnxruntime()
{
#ifdef ENABLE_ONNXRUNTIME
  std::string onnx_path = "../../../hub/onnx/cv/rvm_mobilenetv3_fp32.onnx";
  std::string video_path = "../../../examples/lite/resources/test_lite_rvm_0.mp4";
  std::string background_path = "../../../examples/lite/resources/test_lite_matting_bgr.jpg";
  std::string output_path = "../../../logs/test_lite_rvm_0.mp4";

  cv::Mat background = cv::imread(background_path);
  auto *rvm = new lite::onnxruntime::cv::matting::RobustVideoMatting(onnx_path, 16); // 16 threads
  std::vector<lite::types::MattingContent> contents;

  // 1. video matting.
  rvm->detect_video(video_path, output_path, contents, false, 0.4f,
                    20, true, true, background);

  delete rvm;
#endif
}

static void test_mnn()
{
#ifdef ENABLE_MNN
  std::string mnn_path = "../../../hub/mnn/cv/rvm_mobilenetv3_fp32-1080-1920.mnn";
  std::string video_path = "../../../examples/lite/resources/test_lite_rvm_0.mp4";
  std::string background_path = "../../../examples/lite/resources/test_lite_matting_bgr.jpg";
  std::string output_path = "../../../logs/test_lite_rvm_0_mnn.mp4";

  cv::Mat background = cv::imread(background_path);
  auto *rvm = new lite::mnn::cv::matting::RobustVideoMatting(mnn_path, 16, 0); // 16 threads
  std::vector<lite::types::MattingContent> contents;

  // 1. video matting.
  rvm->detect_video(video_path, output_path, contents, false,
                    20, false, true, background);

  delete rvm;
#endif
}

static void test_ncnn()
{
#ifdef ENABLE_NCNN
  // WARNING: Test Failed!
#endif
}

static void test_tnn()
{
#ifdef ENABLE_TNN

  std::string proto_path = "../../../hub/tnn/cv/rvm_mobilenetv3_fp32-480-480-sim.opt.tnnproto";
  std::string model_path = "../../../hub/tnn/cv/rvm_mobilenetv3_fp32-480-480-sim.opt.tnnmodel";
  std::string video_path = "../../../examples/lite/resources/test_lite_rvm_0.mp4";
  std::string background_path = "../../../examples/lite/resources/test_lite_matting_bgr.jpg";
  std::string output_path = "../../../logs/test_lite_rvm_0_tnn.mp4";

  cv::Mat background = cv::imread(background_path);
  auto *rvm = new lite::tnn::cv::matting::RobustVideoMatting(
      proto_path, model_path, 16); // 16 threads
  std::vector<lite::types::MattingContent> contents;

  // 1. video matting.
  rvm->detect_video(video_path, output_path, contents, false,
                    20, false, true, background);

  delete rvm;
#endif
}

static void test_lite()
{
//  test_default();
//  test_onnxruntime();
  test_mnn();
//  test_ncnn();
//  test_tnn();
}

int main(__unused int argc, __unused char *argv[])
{
  test_lite();
  return 0;
}
