//
// Created by DefTruth on 2022/3/27.
//

#include "lite/lite.h"

static void test_default()
{
  std::string onnx_path = "../../../examples/hub/onnx/cv/quarrying_insect_detector.onnx";
  std::string test_img_path = "../../../examples/lite/resources/test_lite_insect.jpg";
  std::string save_img_path = "../../../examples/logs/test_lite_insectdet.jpg";

  // 1. Test Default Engine ONNXRuntime
  lite::cv::detection::InsectDet *insectdet = new lite::cv::detection::InsectDet(onnx_path); // default

  std::vector<lite::types::Boxf> detected_boxes;
  cv::Mat img_bgr = cv::imread(test_img_path);
  insectdet->detect(img_bgr, detected_boxes);

  lite::utils::draw_boxes_inplace(img_bgr, detected_boxes);

  cv::imwrite(save_img_path, img_bgr);

  std::cout << "Default Version Detected Boxes Num: " << detected_boxes.size() << std::endl;

  delete insectdet;

}

static void test_onnxruntime()
{
#ifdef ENABLE_ONNXRUNTIME
  std::string onnx_path = "../../../examples/hub/onnx/cv/quarrying_insect_detector.onnx";
  std::string test_img_path = "../../../examples/lite/resources/test_lite_insect.jpg";
  std::string save_img_path = "../../../examples/logs/test_lite_insectdet_onnx.jpg";

  // 2. Test Specific Engine ONNXRuntime
  lite::onnxruntime::cv::detection::InsectDet *insectdet =
      new lite::onnxruntime::cv::detection::InsectDet(onnx_path);

  std::vector<lite::types::Boxf> detected_boxes;
  cv::Mat img_bgr = cv::imread(test_img_path);
  insectdet->detect(img_bgr, detected_boxes);

  lite::utils::draw_boxes_inplace(img_bgr, detected_boxes);

  cv::imwrite(save_img_path, img_bgr);

  std::cout << "ONNXRuntime Version Detected Boxes Num: " << detected_boxes.size() << std::endl;

  delete insectdet;
#endif
}

static void test_mnn()
{
#ifdef ENABLE_MNN
  std::string mnn_path = "../../../examples/hub/mnn/cv/quarrying_insect_detector.mnn";
  std::string test_img_path = "../../../examples/lite/resources/test_lite_insect.jpg";
  std::string save_img_path = "../../../examples/logs/test_lite_insectdet_mnn.jpg";

  // 3. Test Specific Engine MNN
  lite::mnn::cv::detection::InsectDet *insectdet =
      new lite::mnn::cv::detection::InsectDet(mnn_path);

  std::vector<lite::types::Boxf> detected_boxes;
  cv::Mat img_bgr = cv::imread(test_img_path);
  insectdet->detect(img_bgr, detected_boxes);

  lite::utils::draw_boxes_inplace(img_bgr, detected_boxes);

  cv::imwrite(save_img_path, img_bgr);

  std::cout << "MNN Version Detected Boxes Num: " << detected_boxes.size() << std::endl;

  delete insectdet;
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
  std::string proto_path = "../../../examples/hub/tnn/cv/quarrying_insect_detector.tnnproto";
  std::string model_path = "../../../examples/hub/tnn/cv/quarrying_insect_detector.tnnmodel";
  std::string test_img_path = "../../../examples/lite/resources/test_lite_insect.jpg";
  std::string save_img_path = "../../../examples/logs/test_lite_insectdet_tnn.jpg";

  // 5. Test Specific Engine TNN
  lite::tnn::cv::detection::InsectDet *insectdet =
      new lite::tnn::cv::detection::InsectDet(proto_path, model_path);

  std::vector<lite::types::Boxf> detected_boxes;
  cv::Mat img_bgr = cv::imread(test_img_path);
  insectdet->detect(img_bgr, detected_boxes);

  lite::utils::draw_boxes_inplace(img_bgr, detected_boxes);

  cv::imwrite(save_img_path, img_bgr);

  std::cout << "TNN Version Detected Boxes Num: " << detected_boxes.size() << std::endl;

  delete insectdet;
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
