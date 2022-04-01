//
// Created by DefTruth on 2022/4/1.
//

#include "lite/lite.h"

static void test_default()
{
  std::string onnx_path = "../../../hub/onnx/cv/modnet_photographic_portrait_matting.onnx";
  std::string test_img_path = "../../../examples/lite/resources/test_lite_matting_input.jpg";
  std::string test_bgr_path = "../../../examples/lite/resources/test_lite_matting_bgr.jpg";
  std::string save_fgr_path = "../../../logs/test_lite_modnet_dyn_fgr.jpg";
  std::string save_pha_path = "../../../logs/test_lite_modnet_dyn_pha.jpg";
  std::string save_merge_path = "../../../logs/test_lite_modnet_dyn_merge.jpg";
  std::string save_swap_path = "../../../logs/test_lite_modnet_dyn_swap.jpg";

  lite::cv::matting::MODNetDyn *modnet_dyn =
      new lite::cv::matting::MODNetDyn(onnx_path, 4); // 4 threads

  lite::types::MattingContent content;
  cv::Mat img_bgr = cv::imread(test_img_path);
  cv::Mat bgr_mat = cv::imread(test_bgr_path);

  // 1. image matting.
  modnet_dyn->detect(img_bgr, content, true);

  if (content.flag)
  {
    if (!content.fgr_mat.empty()) cv::imwrite(save_fgr_path, content.fgr_mat);
    if (!content.pha_mat.empty()) cv::imwrite(save_pha_path, content.pha_mat * 255.);
    if (!content.merge_mat.empty()) cv::imwrite(save_merge_path, content.merge_mat);
    // swap background
    cv::Mat out_mat;
    lite::utils::swap_background(content.fgr_mat, content.pha_mat, bgr_mat, out_mat, true);
    if (!out_mat.empty())
    {
      cv::imwrite(save_swap_path, out_mat);
      std::cout << "Saved Swap Image Done!" << std::endl;
    }

    std::cout << "Default Version MGMatting Done!" << std::endl;
  }

  delete modnet_dyn;
}

static void test_onnxruntime()
{
#ifdef ENABLE_ONNXRUNTIME
  std::string onnx_path = "../../../hub/onnx/cv/modnet_photographic_portrait_matting.onnx";
  std::string test_img_path = "../../../examples/lite/resources/test_lite_matting_input.jpg";
  std::string test_bgr_path = "../../../examples/lite/resources/test_lite_matting_bgr.jpg";
  std::string save_fgr_path = "../../../logs/test_lite_modnet_dyn_fgr_onnx.jpg";
  std::string save_pha_path = "../../../logs/test_lite_modnet_dyn_pha_onnx.jpg";
  std::string save_merge_path = "../../../logs/test_lite_modnet_dyn_merge_onnx.jpg";
  std::string save_swap_path = "../../../logs/test_lite_modnet_dyn_swap_onnx.jpg";

  lite::onnxruntime::cv::matting::MODNetDyn *modnet_dyn =
      new lite::onnxruntime::cv::matting::MODNetDyn(onnx_path, 4); // 4 threads

  lite::types::MattingContent content;
  cv::Mat img_bgr = cv::imread(test_img_path);
  cv::Mat bgr_mat = cv::imread(test_bgr_path);

  // 1. image matting.
  modnet_dyn->detect(img_bgr, content, true);

  if (content.flag)
  {
    if (!content.fgr_mat.empty()) cv::imwrite(save_fgr_path, content.fgr_mat);
    if (!content.pha_mat.empty()) cv::imwrite(save_pha_path, content.pha_mat * 255.);
    if (!content.merge_mat.empty()) cv::imwrite(save_merge_path, content.merge_mat);
    // swap background
    cv::Mat out_mat;
    lite::utils::swap_background(content.fgr_mat, content.pha_mat, bgr_mat, out_mat, true);
    if (!out_mat.empty())
    {
      cv::imwrite(save_swap_path, out_mat);
      std::cout << "Saved Swap Image Done!" << std::endl;
    }

    std::cout << "Default Version MGMatting Done!" << std::endl;
  }

  delete modnet_dyn;
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
