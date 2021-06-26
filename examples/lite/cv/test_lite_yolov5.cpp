//
// Created by DefTruth on 2021/6/21.
//

#include "lite/lite.h"

static void test_default()
{
  std::string onnx_path = "../../../hub/onnx/cv/yolov5s.onnx";
  std::string test_img_path = "../../../examples/lite/resources/test_lite_yolov5_1.jpg";
  std::string save_img_path = "../../../logs/test_lite_yolov5_1.jpg";

  // 1. Test Default Engine ONNXRuntime
  lite::cv::detection::YoloV5 *yolov5 = new lite::cv::detection::YoloV5(onnx_path); // default

  std::vector<lite::cv::types::Boxf> detected_boxes;
  cv::Mat img_bgr = cv::imread(test_img_path);
  yolov5->detect(img_bgr, detected_boxes);

  lite::cv::utils::draw_boxes_inplace(img_bgr, detected_boxes);

  cv::imwrite(save_img_path, img_bgr);

  std::cout << "Default Version Detected Boxes Num: " << detected_boxes.size() << std::endl;

  delete yolov5;

}

static void test_onnxruntime()
{
  std::string onnx_path = "../../../hub/onnx/cv/yolov5s.onnx";
  std::string test_img_path = "../../../examples/lite/resources/test_lite_yolov5_2.jpg";
  std::string save_img_path = "../../../logs/test_lite_yolov5_2.jpg";

  // 2. Test Specific Engine ONNXRuntime
  lite::onnxruntime::cv::detection::YoloV5 *yolov5 =
      new lite::onnxruntime::cv::detection::YoloV5(onnx_path);

  std::vector<lite::onnxruntime::cv::types::Boxf> detected_boxes;
  cv::Mat img_bgr = cv::imread(test_img_path);
  yolov5->detect(img_bgr, detected_boxes);

  lite::onnxruntime::cv::utils::draw_boxes_inplace(img_bgr, detected_boxes);

  cv::imwrite(save_img_path, img_bgr);

  std::cout << "ONNXRuntime Version Detected Boxes Num: " << detected_boxes.size() << std::endl;

  delete yolov5;
}

static void test_mnn()
{

}

static void test_ncnn()
{

}

static void test_tnn()
{

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
