//
// Created by DefTruth on 2022/6/19.
//

#include "lite/lite.h"

static void test_default()
{
  std::string onnx_path = "../../../hub/onnx/cv/ext_portrait_seg_ExtremeC3_224x224.onnx";
  std::string test_img_path = "../../../examples/lite/resources/test_lite_head_seg.png";
  std::string save_img_path = "../../../logs/test_lite_portrait_seg_extremec3net.jpg";

  lite::cv::segmentation::PortraitSegExtremeC3Net *portrait_seg_extremec3net =
      new lite::cv::segmentation::PortraitSegExtremeC3Net(onnx_path, 4); // 4 threads

  lite::types::PortraitSegContent content;
  cv::Mat img_bgr = cv::imread(test_img_path);
  portrait_seg_extremec3net->detect(img_bgr, content);

  if (content.flag)
  {
    if (!content.mask.empty()) cv::imwrite(save_img_path, content.mask * 255.f);
    std::cout << "Default Version PortraitSegExtremeC3Net Done!" << std::endl;
  }

  delete portrait_seg_extremec3net;
}

static void test_onnxruntime()
{
#ifdef ENABLE_ONNXRUNTIME
  std::string onnx_path = "../../../hub/onnx/cv/ext_portrait_seg_ExtremeC3_224x224.onnx";
  std::string test_img_path = "../../../examples/lite/resources/test_lite_head_seg_1.png";
  std::string save_img_path = "../../../logs/test_lite_portrait_seg_extremec3net_1_onnx.jpg";

  lite::onnxruntime::cv::segmentation::PortraitSegExtremeC3Net *portrait_seg_extremec3net =
      new lite::onnxruntime::cv::segmentation::PortraitSegExtremeC3Net(onnx_path, 4); // 4 threads

  lite::types::PortraitSegContent content;
  cv::Mat img_bgr = cv::imread(test_img_path);
  portrait_seg_extremec3net->detect(img_bgr, content);

  if (content.flag)
  {
    if (!content.mask.empty()) cv::imwrite(save_img_path, content.mask * 255.f);
    std::cout << "ONNXRuntime Version PortraitSegExtremeC3Net Done!" << std::endl;
  }

  delete portrait_seg_extremec3net;
#endif
}

static void test_mnn()
{
#ifdef ENABLE_MNN
  std::string mnn_path = "../../../hub/mnn/cv/ext_portrait_seg_ExtremeC3_224x224.mnn";
  std::string test_img_path = "../../../examples/lite/resources/test_lite_head_seg_1.png";
  std::string save_img_path = "../../../logs/test_lite_portrait_seg_extremec3net_1_mnn.jpg";

  lite::mnn::cv::segmentation::PortraitSegExtremeC3Net *portrait_seg_extremec3net =
      new lite::mnn::cv::segmentation::PortraitSegExtremeC3Net(mnn_path, 4); // 4 threads

  lite::types::PortraitSegContent content;
  cv::Mat img_bgr = cv::imread(test_img_path);
  portrait_seg_extremec3net->detect(img_bgr, content);

  if (content.flag)
  {
    if (!content.mask.empty()) cv::imwrite(save_img_path, content.mask * 255.f);
    std::cout << "MNN Version PortraitSegExtremeC3Net Done!" << std::endl;
  }

  delete portrait_seg_extremec3net;
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
