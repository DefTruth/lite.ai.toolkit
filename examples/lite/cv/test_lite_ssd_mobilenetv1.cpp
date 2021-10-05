//
// Created by DefTruth on 2021/6/26.
//

#include "lite/lite.h"

static void test_default()
{
  std::string onnx_path = "../../../hub/onnx/cv/ssd_mobilenet_v1_10.onnx";
  std::string test_img_path = "../../../examples/lite/resources/test_lite_ssd_mobilenetv1.png";
  std::string save_img_path = "../../../logs/test_lite_ssd_mobilenetv1.jpg";

  lite::cv::detection::SSDMobileNetV1 *ssd_mobilenetv1 =
      new lite::cv::detection::SSDMobileNetV1(onnx_path);

  std::vector<lite::cv::types::Boxf> detected_boxes;
  cv::Mat img_bgr = cv::imread(test_img_path);
  ssd_mobilenetv1->detect(img_bgr, detected_boxes);

  lite::cv::utils::draw_boxes_inplace(img_bgr, detected_boxes);

  cv::imwrite(save_img_path, img_bgr);

  std::cout << "Default Version Done! Detected Boxes Num: " << detected_boxes.size() << std::endl;

  delete ssd_mobilenetv1;
}

static void test_onnxruntime()
{
  std::string onnx_path = "../../../hub/onnx/cv/ssd_mobilenet_v1_10.onnx";
  std::string test_img_path = "../../../examples/lite/resources/test_lite_ssd_mobilenetv1.png";
  std::string save_img_path = "../../../logs/test_onnx_ssd_mobilenetv1.jpg";

  lite::onnxruntime::cv::detection::SSDMobileNetV1 *ssd_mobilenetv1 =
      new lite::onnxruntime::cv::detection::SSDMobileNetV1(onnx_path);

  std::vector<lite::onnxruntime::cv::types::Boxf> detected_boxes;
  cv::Mat img_bgr = cv::imread(test_img_path);
  ssd_mobilenetv1->detect(img_bgr, detected_boxes);

  lite::onnxruntime::cv::utils::draw_boxes_inplace(img_bgr, detected_boxes);

  cv::imwrite(save_img_path, img_bgr);

  std::cout << "ONNXRuntime Version Done! Detected Boxes Num: " << detected_boxes.size() << std::endl;

  delete ssd_mobilenetv1;
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
