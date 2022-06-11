//
// Created by DefTruth on 2022/6/11.
//

#include "lite/lite.h"

static void test_default()
{
  std::string onnx_path = "../../../hub/onnx/cv/minivision_head_seg.onnx";
  std::string test_img_path = "../../../examples/lite/resources/test_lite_head_seg.png";
  std::string save_img_path = "../../../logs/test_lite_head_seg.jpg";

  lite::cv::segmentation::HeadSeg *head_seg =
      new lite::cv::segmentation::HeadSeg(onnx_path, 4); // 4 threads

  lite::types::HeadSegContent content;
  cv::Mat img_bgr = cv::imread(test_img_path);
  head_seg->detect(img_bgr, content);

  if (content.flag)
  {
    if (!content.mask.empty()) cv::imwrite(save_img_path, content.mask * 255.f);
    std::cout << "Default Version HeadSeg Done!" << std::endl;
  }

  delete head_seg;
}

static void test_onnxruntime()
{
#ifdef ENABLE_ONNXRUNTIME
  std::string onnx_path = "../../../hub/onnx/cv/minivision_head_seg.onnx";
  std::string test_img_path = "../../../examples/lite/resources/test_lite_head_seg_1.png";
  std::string save_img_path = "../../../logs/test_lite_head_seg_1_onnx.jpg";

  lite::onnxruntime::cv::segmentation::HeadSeg *head_seg =
      new lite::onnxruntime::cv::segmentation::HeadSeg(onnx_path, 4); // 4 threads

  lite::types::HeadSegContent content;
  cv::Mat img_bgr = cv::imread(test_img_path);
  head_seg->detect(img_bgr, content);

  if (content.flag)
  {
    if (!content.mask.empty()) cv::imwrite(save_img_path, content.mask * 255.f);
    std::cout << "ONNXRuntime Version HeadSeg Done!" << std::endl;
  }

  delete head_seg;
#endif
}

static void test_mnn()
{
#ifdef ENABLE_MNN
  std::string mnn_path = "../../../hub/mnn/cv/minivision_head_seg.mnn";
  std::string test_img_path = "../../../examples/lite/resources/test_lite_head_seg_1.png";
  std::string save_img_path = "../../../logs/test_lite_head_seg_1_mnn.jpg";

  lite::mnn::cv::segmentation::HeadSeg *head_seg =
      new lite::mnn::cv::segmentation::HeadSeg(mnn_path, 4); // 4 threads

  lite::types::HeadSegContent content;
  cv::Mat img_bgr = cv::imread(test_img_path);
  head_seg->detect(img_bgr, content);

  if (content.flag)
  {
    if (!content.mask.empty()) cv::imwrite(save_img_path, content.mask * 255.f);
    std::cout << "MNN Version HeadSeg Done!" << std::endl;
  }

  delete head_seg;
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
  // WARN: BAD RESULTS !!!!
  std::string proto_path = "../../../hub/tnn/cv/minivision_head_seg.opt.tnnproto";
  std::string model_path = "../../../hub/tnn/cv/minivision_head_seg.opt.tnnmodel";
  std::string test_img_path = "../../../examples/lite/resources/test_lite_head_seg_1.png";
  std::string save_img_path = "../../../logs/test_lite_head_seg_1_tnn.jpg";

  lite::tnn::cv::segmentation::HeadSeg *head_seg =
      new lite::tnn::cv::segmentation::HeadSeg(proto_path, model_path, 4); // 4 threads

  lite::types::HeadSegContent content;
  cv::Mat img_bgr = cv::imread(test_img_path);
  head_seg->detect(img_bgr, content);

  if (content.flag)
  {
    if (!content.mask.empty()) cv::imwrite(save_img_path, content.mask * 255.f);
    std::cout << "TNN Version HeadSeg Done!" << std::endl;
  }

  delete head_seg;
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
