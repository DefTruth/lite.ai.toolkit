//
// Created by DefTruth on 2022/6/18.
//

#include "lite/lite.h"

static void test_default()
{
  std::string onnx_path = "../../../hub/onnx/cv/fast_portrait_seg_SINet_bi_320_256.onnx";
  std::string test_img_path = "../../../examples/lite/resources/test_lite_head_seg.png";
  std::string save_img_path = "../../../logs/test_lite_fast_portrait_seg.jpg";

  lite::cv::segmentation::FastPortraitSeg *fast_portrait_seg =
      new lite::cv::segmentation::FastPortraitSeg(onnx_path, 4); // 4 threads

  lite::types::PortraitSegContent content;
  cv::Mat img_bgr = cv::imread(test_img_path);
  fast_portrait_seg->detect(img_bgr, content);

  if (content.flag)
  {
    if (!content.mask.empty()) cv::imwrite(save_img_path, content.mask * 255.f);
    std::cout << "Default Version FastPortraitSeg Done!" << std::endl;
  }

  delete fast_portrait_seg;
}

static void test_onnxruntime()
{
#ifdef ENABLE_ONNXRUNTIME
  std::string onnx_path = "../../../hub/onnx/cv/fast_portrait_seg_SINet_bi_320_256.onnx";
  std::string test_img_path = "../../../examples/lite/resources/test_lite_head_seg_1.png";
  std::string save_img_path = "../../../logs/test_lite_fast_portrait_seg_1_onnx.jpg";

  lite::onnxruntime::cv::segmentation::FastPortraitSeg *fast_portrait_seg =
      new lite::onnxruntime::cv::segmentation::FastPortraitSeg(onnx_path, 4); // 4 threads

  lite::types::PortraitSegContent content;
  cv::Mat img_bgr = cv::imread(test_img_path);
  fast_portrait_seg->detect(img_bgr, content);

  if (content.flag)
  {
    if (!content.mask.empty()) cv::imwrite(save_img_path, content.mask * 255.f);
    std::cout << "ONNXRuntime Version FastPortraitSeg Done!" << std::endl;
  }

  delete fast_portrait_seg;
#endif
}

static void test_mnn()
{
#ifdef ENABLE_MNN
  std::string mnn_path = "../../../hub/mnn/cv/fast_portrait_seg_SINet_bi_320_256.mnn";
  std::string test_img_path = "../../../examples/lite/resources/test_lite_head_seg_1.png";
  std::string save_img_path = "../../../logs/test_lite_fast_portrait_seg_1_mnn.jpg";

  lite::mnn::cv::segmentation::FastPortraitSeg *fast_portrait_seg =
      new lite::mnn::cv::segmentation::FastPortraitSeg(mnn_path, 4); // 4 threads

  lite::types::PortraitSegContent content;
  cv::Mat img_bgr = cv::imread(test_img_path);
  fast_portrait_seg->detect(img_bgr, content);

  if (content.flag)
  {
    if (!content.mask.empty()) cv::imwrite(save_img_path, content.mask * 255.f);
    std::cout << "MNN Version FastPortraitSeg Done!" << std::endl;
  }

  delete fast_portrait_seg;
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
