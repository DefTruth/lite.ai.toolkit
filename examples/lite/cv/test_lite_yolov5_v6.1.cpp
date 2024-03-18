//
// Created by DefTruth on 2022/5/8.
//

#include "lite/lite.h"

static void test_default()
{
  std::string onnx_path = "../../../examples/hub/onnx/cv/yolov5s.v6.1.640x640.onnx";
  std::string test_img_path = "../../../examples/lite/resources/test_lite_yolov5_1.jpg";
  std::string save_img_path = "../../../examples/logs/test_lite_yolov5_v6.1_1.jpg";

  // 1. Test Default Engine ONNXRuntime
  lite::cv::detection::YoloV5_V_6_1 *yolov5 = new lite::cv::detection::YoloV5_V_6_1(onnx_path); // default

  std::vector<lite::types::Boxf> detected_boxes;
  cv::Mat img_bgr = cv::imread(test_img_path);
  yolov5->detect(img_bgr, detected_boxes);

  lite::utils::draw_boxes_inplace(img_bgr, detected_boxes);

  cv::imwrite(save_img_path, img_bgr);

  std::cout << "Default Version Detected Boxes Num: " << detected_boxes.size() << std::endl;

  delete yolov5;

}

static void test_onnxruntime()
{
#ifdef ENABLE_ONNXRUNTIME
  std::string onnx_path = "../../../examples/hub/onnx/cv/yolov5s.v6.1.640x640.onnx";
  std::string test_img_path = "../../../examples/lite/resources/test_lite_yolov5_2.jpg";
  std::string save_img_path = "../../../examples/logs/test_lite_yolov5_v6.1_2.jpg";

  // 2. Test Specific Engine ONNXRuntime
  lite::onnxruntime::cv::detection::YoloV5_V_6_1 *yolov5 =
      new lite::onnxruntime::cv::detection::YoloV5_V_6_1(onnx_path);

  std::vector<lite::types::Boxf> detected_boxes;
  cv::Mat img_bgr = cv::imread(test_img_path);
  yolov5->detect(img_bgr, detected_boxes);

  lite::utils::draw_boxes_inplace(img_bgr, detected_boxes);

  cv::imwrite(save_img_path, img_bgr);

  std::cout << "ONNXRuntime Version Detected Boxes Num: " << detected_boxes.size() << std::endl;

  delete yolov5;
#endif
}

static void test_mnn()
{
#ifdef ENABLE_MNN
  std::string mnn_path = "../../../examples/hub/mnn/cv/yolov5s.v6.1.640x640.mnn";
  std::string test_img_path = "../../../examples/lite/resources/test_lite_yolov5_2.jpg";
  std::string save_img_path = "../../../examples/logs/test_lite_yolov5_v6.1_2_mnn.jpg";

  // 3. Test Specific Engine MNN
  lite::mnn::cv::detection::YoloV5_V_6_1 *yolov5 =
      new lite::mnn::cv::detection::YoloV5_V_6_1(mnn_path);

  std::vector<lite::types::Boxf> detected_boxes;
  cv::Mat img_bgr = cv::imread(test_img_path);
  yolov5->detect(img_bgr, detected_boxes);

  lite::utils::draw_boxes_inplace(img_bgr, detected_boxes);

  cv::imwrite(save_img_path, img_bgr);

  std::cout << "MNN Version Detected Boxes Num: " << detected_boxes.size() << std::endl;

  delete yolov5;
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
