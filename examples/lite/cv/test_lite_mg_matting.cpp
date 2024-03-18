//
// Created by DefTruth on 2021/12/5.
//

#include "lite/lite.h"

static void test_default()
{
  std::string onnx_path = "../../../examples/hub/onnx/cv/MGMatting-DIM-100k.onnx";
  std::string test_img_path = "../../../examples/lite/resources/test_lite_mg_matting_input.jpg";
  std::string test_mask_path = "../../../examples/lite/resources/test_lite_mg_matting_mask.png";
  std::string save_fgr_path = "../../../examples/logs/test_lite_mg_matting_fgr.jpg";
  std::string save_pha_path = "../../../examples/logs/test_lite_mg_matting_pha.jpg";
  std::string save_merge_path = "../../../examples/logs/test_lite_mg_matting_merge.jpg";

  lite::cv::matting::MGMatting *mgmatting =
      new lite::cv::matting::MGMatting(onnx_path, 16); // 16 threads

  lite::types::MattingContent content;
  cv::Mat img_bgr = cv::imread(test_img_path);
  cv::Mat mask = cv::imread(test_mask_path, cv::IMREAD_GRAYSCALE);

  // 1. image matting.
  mgmatting->detect(img_bgr, mask, content, true);

  if (content.flag)
  {
    if (!content.fgr_mat.empty()) cv::imwrite(save_fgr_path, content.fgr_mat);
    if (!content.pha_mat.empty()) cv::imwrite(save_pha_path, content.pha_mat * 255.);
    if (!content.merge_mat.empty()) cv::imwrite(save_merge_path, content.merge_mat);
    std::cout << "Default Version MGMatting Done!" << std::endl;
  }

  delete mgmatting;
}

static void test_onnxruntime()
{
#ifdef ENABLE_ONNXRUNTIME
  std::string onnx_path = "../../../examples/hub/onnx/cv/MGMatting-DIM-100k.onnx";
  std::string test_img_path = "../../../examples/lite/resources/test_lite_mg_matting_input.jpg";
  std::string test_mask_path = "../../../examples/lite/resources/test_lite_mg_matting_mask.png";
  std::string save_fgr_path = "../../../examples/logs/test_lite_mg_matting_fgr_onnx.jpg";
  std::string save_pha_path = "../../../examples/logs/test_lite_mg_matting_pha_onnx.jpg";
  std::string save_merge_path = "../../../examples/logs/test_lite_mg_matting_merge_onnx.jpg";

  lite::onnxruntime::cv::matting::MGMatting *mgmatting =
      new lite::onnxruntime::cv::matting::MGMatting(onnx_path, 16); // 16 threads

  lite::types::MattingContent content;
  cv::Mat img_bgr = cv::imread(test_img_path);
  cv::Mat mask = cv::imread(test_mask_path, cv::IMREAD_GRAYSCALE);

  // 1. image matting.
  mgmatting->detect(img_bgr, mask, content);

  if (content.flag)
  {
    if (!content.fgr_mat.empty()) cv::imwrite(save_fgr_path, content.fgr_mat);
    if (!content.pha_mat.empty()) cv::imwrite(save_pha_path, content.pha_mat * 255.);
    if (!content.merge_mat.empty()) cv::imwrite(save_merge_path, content.merge_mat);
    std::cout << "ONNXRuntime Version MGMatting Done!" << std::endl;
  }

  delete mgmatting;
#endif
}

static void test_mnn()
{
#ifdef ENABLE_MNN
  std::string mnn_path = "../../../examples/hub/mnn/cv/MGMatting-DIM-100k.mnn";
  std::string test_img_path = "../../../examples/lite/resources/test_lite_mg_matting_input.jpg";
  std::string test_mask_path = "../../../examples/lite/resources/test_lite_mg_matting_mask.png";
  std::string save_fgr_path = "../../../examples/logs/test_lite_mg_matting_fgr_mnn.jpg";
  std::string save_pha_path = "../../../examples/logs/test_lite_mg_matting_pha_mnn.jpg";
  std::string save_merge_path = "../../../examples/logs/test_lite_mg_matting_merge_mnn.jpg";

  lite::mnn::cv::matting::MGMatting *mgmatting =
      new lite::mnn::cv::matting::MGMatting(mnn_path, 16); // 16 threads

  lite::types::MattingContent content;
  cv::Mat img_bgr = cv::imread(test_img_path);
  cv::Mat mask = cv::imread(test_mask_path, cv::IMREAD_GRAYSCALE);

  // 1. image matting.
  mgmatting->detect(img_bgr, mask, content, true);

  if (content.flag)
  {
    if (!content.fgr_mat.empty()) cv::imwrite(save_fgr_path, content.fgr_mat);
    if (!content.pha_mat.empty()) cv::imwrite(save_pha_path, content.pha_mat * 255.);
    if (!content.merge_mat.empty()) cv::imwrite(save_merge_path, content.merge_mat);
    std::cout << "MNN Version MGMatting Done!" << std::endl;
  }

  delete mgmatting;
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
  std::string proto_path = "../../../examples/hub/tnn/cv/MGMatting-DIM-100k.opt.tnnproto";
  std::string model_path = "../../../examples/hub/tnn/cv/MGMatting-DIM-100k.opt.tnnmodel";
  std::string test_img_path = "../../../examples/lite/resources/test_lite_mg_matting_input.jpg";
  std::string test_mask_path = "../../../examples/lite/resources/test_lite_mg_matting_mask.png";
  std::string save_fgr_path = "../../../examples/logs/test_lite_mg_matting_fgr_tnn.jpg";
  std::string save_pha_path = "../../../examples/logs/test_lite_mg_matting_pha_tnn.jpg";
  std::string save_merge_path = "../../../examples/logs/test_lite_mg_matting_merge_tnn.jpg";

  lite::tnn::cv::matting::MGMatting *mgmatting =
      new lite::tnn::cv::matting::MGMatting(proto_path, model_path, 16); // 16 threads

  lite::types::MattingContent content;
  cv::Mat img_bgr = cv::imread(test_img_path);
  cv::Mat mask = cv::imread(test_mask_path, cv::IMREAD_GRAYSCALE);

  // 1. image matting.
  mgmatting->detect(img_bgr, mask, content, true);

  if (content.flag)
  {
    if (!content.fgr_mat.empty()) cv::imwrite(save_fgr_path, content.fgr_mat);
    if (!content.pha_mat.empty()) cv::imwrite(save_pha_path, content.pha_mat * 255.);
    if (!content.merge_mat.empty()) cv::imwrite(save_merge_path, content.merge_mat);
    std::cout << "TNN Version MGMatting Done!" << std::endl;
  }

  delete mgmatting;
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
