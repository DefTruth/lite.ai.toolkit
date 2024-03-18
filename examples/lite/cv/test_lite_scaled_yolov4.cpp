//
// Created by DefTruth on 2021/8/8.
//

#include "lite/lite.h"

static void test_default()
{
  std::string onnx_path = "../../../examples/hub/onnx/cv/ScaledYoloV4_yolov4-p5-640-640.onnx";
  std::string test_img_path = "../../../examples/lite/resources/test_lite_detection_1.jpg";
  std::string save_img_path = "../../../examples/logs/test_lite_scaled_yolov4_1.jpg";

  // 1. Test Default Engine ONNXRuntime
  lite::cv::detection::ScaledYoloV4 *scaled_yolov4 = new lite::cv::detection::ScaledYoloV4(onnx_path, 8); // default

  std::vector<lite::types::Boxf> detected_boxes;
  cv::Mat img_bgr = cv::imread(test_img_path);
  scaled_yolov4->detect(img_bgr, detected_boxes);

  lite::utils::draw_boxes_inplace(img_bgr, detected_boxes);

  cv::imwrite(save_img_path, img_bgr);

  std::cout << "Default Version Detected Boxes Num: " << detected_boxes.size() << std::endl;

  delete scaled_yolov4;

}

static void test_onnxruntime()
{
#ifdef ENABLE_ONNXRUNTIME
  std::string onnx_path = "../../../examples/hub/onnx/cv/ScaledYoloV4_yolov4-p5-640-640.onnx";
  std::string test_img_path = "../../../examples/lite/resources/test_lite_detection_2.jpg";
  std::string save_img_path = "../../../examples/logs/test_lite_scaled_yolov4_2.jpg";

  // 2. Test Specific Engine ONNXRuntime
  lite::onnxruntime::cv::detection::ScaledYoloV4 *scaled_yolov4 =
      new lite::onnxruntime::cv::detection::ScaledYoloV4(onnx_path, 8);

  std::vector<lite::types::Boxf> detected_boxes;
  cv::Mat img_bgr = cv::imread(test_img_path);
  scaled_yolov4->detect(img_bgr, detected_boxes);

  lite::utils::draw_boxes_inplace(img_bgr, detected_boxes);

  cv::imwrite(save_img_path, img_bgr);

  std::cout << "ONNXRuntime Version Detected Boxes Num: " << detected_boxes.size() << std::endl;

  delete scaled_yolov4;
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
