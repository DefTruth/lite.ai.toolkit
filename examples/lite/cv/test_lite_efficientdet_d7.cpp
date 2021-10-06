//
// Created by DefTruth on 2021/8/15.
//

#include "lite/lite.h"

static void test_default()
{
  std::string onnx_path = "../../../hub/onnx/cv/efficientdet-d7.onnx";
  std::string test_img_path = "../../../examples/lite/resources/test_lite_detection_1.jpg";
  std::string save_img_path = "../../../logs/test_lite_efficientdet_d7_1.jpg";

  // 1. Test Default Engine ONNXRuntime
  lite::cv::detection::EfficientDetD7 *efficientdet_d7 =
      new lite::cv::detection::EfficientDetD7(onnx_path, 8); // default

  std::vector<lite::cv::types::Boxf> detected_boxes;
  cv::Mat img_bgr = cv::imread(test_img_path);
  efficientdet_d7->detect(img_bgr, detected_boxes);

  lite::cv::utils::draw_boxes_inplace(img_bgr, detected_boxes);

  cv::imwrite(save_img_path, img_bgr);

  std::cout << "Default Version Detected Boxes Num: " << detected_boxes.size() << std::endl;

  delete efficientdet_d7;

}

static void test_onnxruntime()
{
#ifdef ENABLE_ONNXRUNTIME
  std::string onnx_path = "../../../hub/onnx/cv/efficientdet-d7.onnx";
  std::string test_img_path = "../../../examples/lite/resources/test_lite_detection_2.jpg";
  std::string save_img_path = "../../../logs/test_onnx_efficientdet_d7_2.jpg";

  // 2. Test Specific Engine ONNXRuntime
  lite::onnxruntime::cv::detection::EfficientDetD7 *efficientdet_d7 =
      new lite::onnxruntime::cv::detection::EfficientDetD7(onnx_path, 8);

  std::vector<lite::onnxruntime::cv::types::Boxf> detected_boxes;
  cv::Mat img_bgr = cv::imread(test_img_path);
  efficientdet_d7->detect(img_bgr, detected_boxes);

  lite::onnxruntime::cv::utils::draw_boxes_inplace(img_bgr, detected_boxes);

  cv::imwrite(save_img_path, img_bgr);

  std::cout << "ONNXRuntime Version Detected Boxes Num: " << detected_boxes.size() << std::endl;

  delete efficientdet_d7;
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
