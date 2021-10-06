//
// Created by DefTruth on 2021/10/5.
//

#include "lite/lite.h"

static void test_default()
{
  std::string onnx_path = "../../../hub/onnx/cv/nanodet-EfficientNet-Lite2_512.onnx";
  std::string test_img_path = "../../../examples/lite/resources/test_lite_detection_1.jpg";
  std::string save_img_path = "../../../logs/test_lite_nanodet_efficientnet_lite_1.jpg";

  // 1. Test Default Engine ONNXRuntime
  lite::cv::detection::NanoDetEfficientNetLite *nanodet =
      new lite::cv::detection::NanoDetEfficientNetLite(onnx_path); // default

  std::vector<lite::cv::types::Boxf> detected_boxes;
  cv::Mat img_bgr = cv::imread(test_img_path);
  nanodet->detect(img_bgr, detected_boxes);

  lite::cv::utils::draw_boxes_inplace(img_bgr, detected_boxes);

  cv::imwrite(save_img_path, img_bgr);

  std::cout << "Default Version Detected Boxes Num: " << detected_boxes.size() << std::endl;

  delete nanodet;

}

static void test_onnxruntime()
{
#ifdef ENABLE_ONNXRUNTIME
  std::string onnx_path = "../../../hub/onnx/cv/nanodet-EfficientNet-Lite2_512.onnx";
  std::string test_img_path = "../../../examples/lite/resources/test_lite_detection_2.jpg";
  std::string save_img_path = "../../../logs/test_lite_nanodet_efficientnet_lite_2.jpg";

  // 2. Test Specific Engine ONNXRuntime
  lite::onnxruntime::cv::detection::NanoDetEfficientNetLite *nanodet =
      new lite::onnxruntime::cv::detection::NanoDetEfficientNetLite(onnx_path);

  std::vector<lite::onnxruntime::cv::types::Boxf> detected_boxes;
  cv::Mat img_bgr = cv::imread(test_img_path);
  nanodet->detect(img_bgr, detected_boxes);

  lite::onnxruntime::cv::utils::draw_boxes_inplace(img_bgr, detected_boxes);

  cv::imwrite(save_img_path, img_bgr);

  std::cout << "ONNXRuntime Version Detected Boxes Num: " << detected_boxes.size() << std::endl;

  delete nanodet;
#endif
}

static void test_mnn()
{
#ifdef ENABLE_MNN
  std::string mnn_path = "../../../hub/mnn/cv/nanodet-EfficientNet-Lite2_512.mnn";
  std::string test_img_path = "../../../examples/lite/resources/test_lite_detection_2.jpg";
  std::string save_img_path = "../../../logs/test_lite_nanodet_efficientnet_lite_mnn_2.jpg";

  // 2. Test Specific Engine MNN
  lite::mnn::cv::detection::NanoDetEfficientNetLite *nanodet =
      new lite::mnn::cv::detection::NanoDetEfficientNetLite(mnn_path);

  std::vector<lite::mnn::cv::types::Boxf> detected_boxes;
  cv::Mat img_bgr = cv::imread(test_img_path);
  nanodet->detect(img_bgr, detected_boxes);

  lite::mnn::cv::utils::draw_boxes_inplace(img_bgr, detected_boxes);
  cv::imwrite(save_img_path, img_bgr);

  std::cout << "MNN Version Detected Boxes Num: " << detected_boxes.size() << std::endl;

  delete nanodet;
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
