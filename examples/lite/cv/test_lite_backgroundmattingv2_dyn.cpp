//
// Created by DefTruth on 2022/4/10.
//

#include "lite/lite.h"

static void test_default()
{
  std::string onnx_path = "../../../hub/onnx/cv/BGMv2_mobilenetv2_hd_dynamic.onnx";
  std::string test_src_path = "../../../examples/lite/resources/test_lite_bgmv2_src.png";
  std::string test_bgr_path = "../../../examples/lite/resources/test_lite_bgmv2_bgr.png";
  std::string save_fgr_path = "../../../logs/test_lite_bgmv2_dyn_fgr.jpg";
  std::string save_pha_path = "../../../logs/test_lite_bgmv2_dyn_pha.jpg";
  std::string save_merge_path = "../../../logs/test_lite_bgmv2_dyn_merge.jpg";

  lite::cv::matting::BackgroundMattingV2Dyn *bgmv2_dyn =
      new lite::cv::matting::BackgroundMattingV2Dyn(onnx_path, 16); // 16 threads

  lite::types::MattingContent content;
  cv::Mat src = cv::imread(test_src_path);
  cv::Mat bgr = cv::imread(test_bgr_path);

  // 1. image matting.
  bgmv2_dyn->detect(src, bgr, content, true);

  if (content.flag)
  {
    if (!content.fgr_mat.empty()) cv::imwrite(save_fgr_path, content.fgr_mat);
    if (!content.pha_mat.empty()) cv::imwrite(save_pha_path, content.pha_mat * 255.);
    if (!content.merge_mat.empty()) cv::imwrite(save_merge_path, content.merge_mat);
    std::cout << "Default Version BackgroundMattingV2Dyn Done!" << std::endl;
  }

  delete bgmv2_dyn;
}

static void test_onnxruntime()
{
#ifdef ENABLE_ONNXRUNTIME
  std::string onnx_path = "../../../hub/onnx/cv/BGMv2_mobilenetv2_hd_dynamic.onnx";
  std::string test_src_path = "../../../examples/lite/resources/test_lite_bgmv2_src.png";
  std::string test_bgr_path = "../../../examples/lite/resources/test_lite_bgmv2_bgr.png";
  std::string save_fgr_path = "../../../logs/test_lite_bgmv2_dyn_fgr_onnx.jpg";
  std::string save_pha_path = "../../../logs/test_lite_bgmv2_dyn_pha_onnx.jpg";
  std::string save_merge_path = "../../../logs/test_lite_bgmv2_dyn_merge_onnx.jpg";

  lite::onnxruntime::cv::matting::BackgroundMattingV2Dyn *bgmv2_dyn =
      new lite::onnxruntime::cv::matting::BackgroundMattingV2Dyn(onnx_path, 16); // 16 threads

  lite::types::MattingContent content;
  cv::Mat src = cv::imread(test_src_path);
  cv::Mat bgr = cv::imread(test_bgr_path);

  // 1. image matting.
  bgmv2_dyn->detect(src, bgr, content, true);

  if (content.flag)
  {
    if (!content.fgr_mat.empty()) cv::imwrite(save_fgr_path, content.fgr_mat);
    if (!content.pha_mat.empty()) cv::imwrite(save_pha_path, content.pha_mat * 255.);
    if (!content.merge_mat.empty()) cv::imwrite(save_merge_path, content.merge_mat);
    std::cout << "ONNXRuntime Version BackgroundMattingV2Dyn Done!" << std::endl;
  }

  delete bgmv2_dyn;
#endif
}

static void test_mnn()
{
#ifdef ENABLE_MNN
#endif
}

static void test_ncnn()
{
#ifdef ENABLE_NCNN
#endif
}

static void test_tnn()
{
#ifdef ENABLE_TNN
#endif
}

static void test_lite()
{
  test_default();
  test_onnxruntime();
  test_mnn();
  test_ncnn();
  test_tnn();
}

int main(__unused int argc, __unused char *argv[])
{
  test_lite();
  return 0;
}
