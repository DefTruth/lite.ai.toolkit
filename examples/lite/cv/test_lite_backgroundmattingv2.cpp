//
// Created by DefTruth on 2022/4/10.
//

#include "lite/lite.h"

static void test_default()
{
  std::string onnx_path = "../../../examples/hub/onnx/cv/BGMv2_mobilenetv2-512x512-full.onnx";
  std::string test_src_path = "../../../examples/lite/resources/test_lite_bgmv2_src.png";
  std::string test_bgr_path = "../../../examples/lite/resources/test_lite_bgmv2_bgr.png";
  std::string save_fgr_path = "../../../examples/logs/test_lite_bgmv2_fgr.jpg";
  std::string save_pha_path = "../../../examples/logs/test_lite_bgmv2_pha.jpg";
  std::string save_merge_path = "../../../examples/logs/test_lite_bgmv2_merge.jpg";

  lite::cv::matting::BackgroundMattingV2 *bgmv2 =
      new lite::cv::matting::BackgroundMattingV2(onnx_path, 16); // 16 threads

  lite::types::MattingContent content;
  cv::Mat src = cv::imread(test_src_path);
  cv::Mat bgr = cv::imread(test_bgr_path);

  // 1. image matting.
  bgmv2->detect(src, bgr, content, true);

  if (content.flag)
  {
    if (!content.fgr_mat.empty()) cv::imwrite(save_fgr_path, content.fgr_mat);
    if (!content.pha_mat.empty()) cv::imwrite(save_pha_path, content.pha_mat * 255.);
    if (!content.merge_mat.empty()) cv::imwrite(save_merge_path, content.merge_mat);
    std::cout << "Default Version BackgroundMattingV2 Done!" << std::endl;
  }

  delete bgmv2;
}

static void test_onnxruntime()
{
#ifdef ENABLE_ONNXRUNTIME
  std::string onnx_path = "../../../examples/hub/onnx/cv/BGMv2_mobilenetv2-512x512-full.onnx";
  std::string test_src_path = "../../../examples/lite/resources/test_lite_bgmv2_src.png";
  std::string test_bgr_path = "../../../examples/lite/resources/test_lite_bgmv2_bgr.png";
  std::string save_fgr_path = "../../../examples/logs/test_lite_bgmv2_fgr_onnx.jpg";
  std::string save_pha_path = "../../../examples/logs/test_lite_bgmv2_pha_onnx.jpg";
  std::string save_merge_path = "../../../examples/logs/test_lite_bgmv2_merge_onnx.jpg";

  lite::onnxruntime::cv::matting::BackgroundMattingV2 *bgmv2 =
      new lite::onnxruntime::cv::matting::BackgroundMattingV2(onnx_path, 16); // 16 threads

  lite::types::MattingContent content;
  cv::Mat src = cv::imread(test_src_path);
  cv::Mat bgr = cv::imread(test_bgr_path);

  // 1. image matting.
  bgmv2->detect(src, bgr, content, true);

  if (content.flag)
  {
    if (!content.fgr_mat.empty()) cv::imwrite(save_fgr_path, content.fgr_mat);
    if (!content.pha_mat.empty()) cv::imwrite(save_pha_path, content.pha_mat * 255.);
    if (!content.merge_mat.empty()) cv::imwrite(save_merge_path, content.merge_mat);
    std::cout << "ONNXRuntime Version BackgroundMattingV2 Done!" << std::endl;
  }

  delete bgmv2;
#endif
}

static void test_mnn()
{
#ifdef ENABLE_MNN
  std::string mnn_path = "../../../examples/hub/mnn/cv/BGMv2_mobilenetv2-512x512-full.mnn";
  std::string test_src_path = "../../../examples/lite/resources/test_lite_bgmv2_src.png";
  std::string test_bgr_path = "../../../examples/lite/resources/test_lite_bgmv2_bgr.png";
  std::string save_fgr_path = "../../../examples/logs/test_lite_bgmv2_fgr_mnn.jpg";
  std::string save_pha_path = "../../../examples/logs/test_lite_bgmv2_pha_mnn.jpg";
  std::string save_merge_path = "../../../examples/logs/test_lite_bgmv2_merge_mnn.jpg";

  lite::mnn::cv::matting::BackgroundMattingV2 *bgmv2 =
      new lite::mnn::cv::matting::BackgroundMattingV2(mnn_path, 16); // 16 threads

  lite::types::MattingContent content;
  cv::Mat src = cv::imread(test_src_path);
  cv::Mat bgr = cv::imread(test_bgr_path);

  // 1. image matting.
  bgmv2->detect(src, bgr, content, true);

  if (content.flag)
  {
    if (!content.fgr_mat.empty()) cv::imwrite(save_fgr_path, content.fgr_mat);
    if (!content.pha_mat.empty()) cv::imwrite(save_pha_path, content.pha_mat * 255.);
    if (!content.merge_mat.empty()) cv::imwrite(save_merge_path, content.merge_mat);
    std::cout << "MNN Version BackgroundMattingV2 Done!" << std::endl;
  }

  delete bgmv2;
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
  std::string proto_path = "../../../examples/hub/tnn/cv/BGMv2_mobilenetv2-512x512-full.opt.tnnproto";
  std::string model_path = "../../../examples/hub/tnn/cv/BGMv2_mobilenetv2-512x512-full.opt.tnnmodel";
  std::string test_src_path = "../../../examples/lite/resources/test_lite_bgmv2_src.png";
  std::string test_bgr_path = "../../../examples/lite/resources/test_lite_bgmv2_bgr.png";
  std::string save_fgr_path = "../../../examples/logs/test_lite_bgmv2_fgr_tnn.jpg";
  std::string save_pha_path = "../../../examples/logs/test_lite_bgmv2_pha_tnn.jpg";
  std::string save_merge_path = "../../../examples/logs/test_lite_bgmv2_merge_tnn.jpg";

  lite::tnn::cv::matting::BackgroundMattingV2 *bgmv2 =
      new lite::tnn::cv::matting::BackgroundMattingV2(proto_path, model_path, 16); // 16 threads

  lite::types::MattingContent content;
  cv::Mat src = cv::imread(test_src_path);
  cv::Mat bgr = cv::imread(test_bgr_path);

  // 1. image matting.
  bgmv2->detect(src, bgr, content, true);

  if (content.flag)
  {
    if (!content.fgr_mat.empty()) cv::imwrite(save_fgr_path, content.fgr_mat);
    if (!content.pha_mat.empty()) cv::imwrite(save_pha_path, content.pha_mat * 255.);
    if (!content.merge_mat.empty()) cv::imwrite(save_merge_path, content.merge_mat);
    std::cout << "TNN Version BackgroundMattingV2 Done!" << std::endl;
  }

  delete bgmv2;
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
