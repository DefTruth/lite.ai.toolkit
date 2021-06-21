//
// Created by DefTruth on 2021/6/21.
//

#include "lite/lite.h"

static void test_lite_yolov5()
{
  std::string onnx_path = "../../../hub/onnx/cv/yolov5s.onnx";
  std::string test_img_path_1 = "../../../examples/lite/resources/test_lite_yolov5_1.jpg";
  std::string test_img_path_2 = "../../../examples/lite/resources/test_lite_yolov5_2.jpg";
  std::string save_img_path_1 = "../../../logs/test_lite_yolov5_1.jpg";
  std::string save_img_path_2 = "../../../logs/test_lite_yolov5_2.jpg";

  // 1. Test Default Engine ONNXRuntime
  lite::cv::detection::YoloV5 *yolov5 = new lite::cv::detection::YoloV5(onnx_path); // default

  std::vector<lite::cv::types::Boxf> detected_boxes_1;
  cv::Mat img_bgr_1 = cv::imread(test_img_path_1);
  yolov5->detect(img_bgr_1, detected_boxes_1);

  lite::cv::utils::draw_boxes_inplace(img_bgr_1, detected_boxes_1);

  cv::imwrite(save_img_path_1, img_bgr_1);

  std::cout << "Default Version Detected Boxes Num: " << detected_boxes_1.size() << std::endl;

  delete yolov5;

  // 2. Test Specific Engine ONNXRuntime
  lite::onnxruntime::cv::detection::YoloV5 *onnx_yolov5 = new lite::onnxruntime::cv::detection::YoloV5(onnx_path);

  std::vector<lite::onnxruntime::cv::types::Boxf> detected_boxes_2;
  cv::Mat img_bgr_2 = cv::imread(test_img_path_2);
  onnx_yolov5->detect(img_bgr_2, detected_boxes_2);

  lite::onnxruntime::cv::utils::draw_boxes_inplace(img_bgr_2, detected_boxes_2);

  cv::imwrite(save_img_path_2, img_bgr_2);

  std::cout << "ONNXRuntime Version Detected Boxes Num: " << detected_boxes_2.size() << std::endl;

  delete onnx_yolov5;
}

int main(__unused int argc, __unused char *argv[])
{
  test_lite_yolov5();
  return 0;
}
