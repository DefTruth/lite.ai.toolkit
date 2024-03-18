//
// Created by DefTruth on 2021/6/26.
//

#include "lite/lite.h"

static void test_default()
{
  std::string onnx_path = "../../../examples/hub/onnx/cv/voc-mobilenetv2-yolov4-640.onnx";
  std::string test_img_path = "../../../examples/lite/resources/test_lite_yolov4.jpg";
  std::string save_img_path = "../../../examples/logs/test_lite_yolov4.jpg";

  lite::cv::detection::YoloV4 *yolov4 = new lite::cv::detection::YoloV4(onnx_path);

  std::vector<lite::types::Boxf> detected_boxes;
  cv::Mat img_bgr = cv::imread(test_img_path);
  yolov4->detect(img_bgr, detected_boxes);

  lite::utils::draw_boxes_inplace(img_bgr, detected_boxes);

  cv::imwrite(save_img_path, img_bgr);

  std::cout << "Default Version Done! Detected Boxes Num: " << detected_boxes.size() << std::endl;

  delete yolov4;
}

static void test_onnxruntime()
{
#ifdef ENABLE_ONNXRUNTIME
  std::string onnx_path = "../../../examples/hub/onnx/cv/voc-mobilenetv2-yolov4-640.onnx";
  std::string test_img_path = "../../../examples/lite/resources/test_lite_yolov4.jpg";
  std::string save_img_path = "../../../examples/logs/test_onnx_yolov4.jpg";

  lite::onnxruntime::cv::detection::YoloV4 *yolov4 =
      new lite::onnxruntime::cv::detection::YoloV4(onnx_path);

  std::vector<lite::types::Boxf> detected_boxes;
  cv::Mat img_bgr = cv::imread(test_img_path);
  yolov4->detect(img_bgr, detected_boxes);

  lite::utils::draw_boxes_inplace(img_bgr, detected_boxes);

  cv::imwrite(save_img_path, img_bgr);

  std::cout << "ONNXRuntime Version Done! Detected Boxes Num: " << detected_boxes.size() << std::endl;

  delete yolov4;
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
