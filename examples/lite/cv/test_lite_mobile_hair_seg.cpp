//
// Created by DefTruth on 2022/6/25.
//

#include "lite/lite.h"

static void test_default()
{
  std::string onnx_path = "../../../examples/hub/onnx/cv/mobile_hair_seg_hairmattenetv1_224x224.onnx";
  std::string test_img_path = "../../../examples/lite/resources/test_lite_face_hair.png";
  std::string save_img_path = "../../../examples/logs/test_lite_mobile_hair_seg.jpg";

  lite::cv::segmentation::MobileHairSeg *mobile_hair_seg =
      new lite::cv::segmentation::MobileHairSeg(onnx_path, 4); // 4 threads

  lite::types::HairSegContent content;
  cv::Mat img_bgr = cv::imread(test_img_path);
  mobile_hair_seg->detect(img_bgr, content);

  if (content.flag)
  {
    if (!content.mask.empty()) cv::imwrite(save_img_path, content.mask * 255.f);
    std::cout << "Default Version MobileHairSeg Done!" << std::endl;
  }

  delete mobile_hair_seg;
}

static void test_onnxruntime()
{
#ifdef ENABLE_ONNXRUNTIME
  std::string onnx_path = "../../../examples/hub/onnx/cv/mobile_hair_seg_hairmattenetv1_224x224.onnx";
  std::string test_img_path = "../../../examples/lite/resources/test_lite_face_hair.png";
  std::string save_img_path = "../../../examples/logs/test_lite_mobile_hair_seg_onnx.jpg";

  lite::onnxruntime::cv::segmentation::MobileHairSeg *mobile_hair_seg =
      new lite::onnxruntime::cv::segmentation::MobileHairSeg(onnx_path, 4); // 4 threads

  lite::types::HairSegContent content;
  cv::Mat img_bgr = cv::imread(test_img_path);
  mobile_hair_seg->detect(img_bgr, content);

  if (content.flag)
  {
    if (!content.mask.empty()) cv::imwrite(save_img_path, content.mask * 255.f);
    std::cout << "ONNXRuntime Version MobileHairSeg Done!" << std::endl;
  }

  delete mobile_hair_seg;
#endif
}

static void test_mnn()
{
#ifdef ENABLE_MNN
  std::string mnn_path = "../../../examples/hub/mnn/cv/mobile_hair_seg_hairmattenetv1_224x224.mnn";
  std::string test_img_path = "../../../examples/lite/resources/test_lite_face_hair.png";
  std::string save_img_path = "../../../examples/logs/test_lite_mobile_hair_seg_mnn.jpg";

  lite::mnn::cv::segmentation::MobileHairSeg *mobile_hair_seg =
      new lite::mnn::cv::segmentation::MobileHairSeg(mnn_path, 4); // 4 threads

  lite::types::HairSegContent content;
  cv::Mat img_bgr = cv::imread(test_img_path);
  mobile_hair_seg->detect(img_bgr, content);

  if (content.flag)
  {
    if (!content.mask.empty()) cv::imwrite(save_img_path, content.mask * 255.f);
    std::cout << "MNN Version MobileHairSeg Done!" << std::endl;
  }

  delete mobile_hair_seg;
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
