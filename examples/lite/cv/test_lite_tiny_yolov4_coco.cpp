//
// Created by DefTruth on 2021/8/7.
//

#include "lite/lite.h"

static void test_default()
{
  std::string onnx_path = "../../../hub/onnx/cv/yolov4_tiny_weights_coco.onnx";
  std::string test_img_path = "../../../examples/lite/resources/test_lite_detection_1.jpg";
  std::string save_img_path = "../../../logs/test_lite_tiny_yolov4_coco_1.jpg";

  // 1. Test Default Engine ONNXRuntime
  lite::cv::detection::TinyYoloV4COCO *tiny_yolov4_coco = new lite::cv::detection::TinyYoloV4COCO(onnx_path); // default

  std::vector<lite::cv::types::Boxf> detected_boxes;
  cv::Mat img_bgr = cv::imread(test_img_path);
  tiny_yolov4_coco->detect(img_bgr, detected_boxes);

  lite::cv::utils::draw_boxes_inplace(img_bgr, detected_boxes);

  cv::imwrite(save_img_path, img_bgr);

  std::cout << "Default Version Detected Boxes Num: " << detected_boxes.size() << std::endl;

  delete tiny_yolov4_coco;

}

static void test_onnxruntime()
{
#ifdef ENABLE_ONNXRUNTIME
  std::string onnx_path = "../../../hub/onnx/cv/yolov4_tiny_weights_coco.onnx";
  std::string test_img_path = "../../../examples/lite/resources/test_lite_detection_2.jpg";
  std::string save_img_path = "../../../logs/test_lite_tiny_yolov4_coco_2.jpg";

  // 2. Test Specific Engine ONNXRuntime
  lite::onnxruntime::cv::detection::TinyYoloV4COCO *tiny_yolov4_coco =
      new lite::onnxruntime::cv::detection::TinyYoloV4COCO(onnx_path);

  std::vector<lite::onnxruntime::cv::types::Boxf> detected_boxes;
  cv::Mat img_bgr = cv::imread(test_img_path);
  tiny_yolov4_coco->detect(img_bgr, detected_boxes);

  lite::onnxruntime::cv::utils::draw_boxes_inplace(img_bgr, detected_boxes);

  cv::imwrite(save_img_path, img_bgr);

  std::cout << "ONNXRuntime Version Detected Boxes Num: " << detected_boxes.size() << std::endl;

  delete tiny_yolov4_coco;
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
