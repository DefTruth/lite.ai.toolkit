//
// Created by DefTruth on 2021/6/21.
//

#include "lite/lite.h"

static void test_default()
{
  std::string onnx_path = "../../../examples/hub/onnx/cv/yolov5s.onnx";
  std::string test_img_path = "../../../examples/lite/resources/test_lite_yolov5_1.jpg";
  std::string save_img_path = "../../../examples/logs/test_lite_yolov5_1.jpg";

  // 1. Test Default Engine ONNXRuntime
  lite::cv::detection::YoloV5 *yolov5 = new lite::cv::detection::YoloV5(onnx_path); // default

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
  std::string onnx_path = "../../../examples/hub/onnx/cv/yolov5s.onnx";
  std::string test_img_path = "../../../examples/lite/resources/test_lite_yolov5_2.jpg";
  std::string save_img_path = "../../../examples/logs/test_lite_yolov5_2.jpg";

  // 2. Test Specific Engine ONNXRuntime
  lite::onnxruntime::cv::detection::YoloV5 *yolov5 =
      new lite::onnxruntime::cv::detection::YoloV5(onnx_path);

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
  std::string mnn_path = "../../../examples/hub/mnn/cv/yolov5s.mnn";
  std::string test_img_path = "../../../examples/lite/resources/test_lite_yolov5_2.jpg";
  std::string save_img_path = "../../../examples/logs/test_lite_yolov5_2_mnn.jpg";

  // 3. Test Specific Engine MNN
  lite::mnn::cv::detection::YoloV5 *yolov5 =
      new lite::mnn::cv::detection::YoloV5(mnn_path);

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
  std::string param_path = "../../../examples/hub/ncnn/cv/yolov5s.opt.param";
  std::string bin_path = "../../../examples/hub/ncnn/cv/yolov5s.opt.bin";
  std::string test_img_path = "../../../examples/lite/resources/test_lite_yolov5_2.jpg";
  std::string save_img_path = "../../../examples/logs/test_lite_yolov5_2_ncnn.jpg";

  // 4. Test Specific Engine NCNN
  lite::ncnn::cv::detection::YoloV5 *yolov5 =
      new lite::ncnn::cv::detection::YoloV5(param_path, bin_path);

  std::vector<lite::types::Boxf> detected_boxes;
  cv::Mat img_bgr = cv::imread(test_img_path);
  yolov5->detect(img_bgr, detected_boxes);

  lite::utils::draw_boxes_inplace(img_bgr, detected_boxes);

  cv::imwrite(save_img_path, img_bgr);

  std::cout << "NCNN Version Detected Boxes Num: " << detected_boxes.size() << std::endl;

  delete yolov5;
#endif
}

static void test_tnn()
{
#ifdef ENABLE_TNN
  std::string proto_path = "../../../examples/hub/tnn/cv/yolov5s.opt.tnnproto";
  std::string model_path = "../../../examples/hub/tnn/cv/yolov5s.opt.tnnmodel";
  std::string test_img_path = "../../../examples/lite/resources/test_lite_yolov5_2.jpg";
  std::string save_img_path = "../../../examples/logs/test_lite_yolov5_2_tnn.jpg";

  // 5. Test Specific Engine TNN
  lite::tnn::cv::detection::YoloV5 *yolov5 =
      new lite::tnn::cv::detection::YoloV5(proto_path, model_path);

  std::vector<lite::types::Boxf> detected_boxes;
  cv::Mat img_bgr = cv::imread(test_img_path);
  yolov5->detect(img_bgr, detected_boxes);

  lite::utils::draw_boxes_inplace(img_bgr, detected_boxes);

  cv::imwrite(save_img_path, img_bgr);

  std::cout << "TNN Version Detected Boxes Num: " << detected_boxes.size() << std::endl;

  delete yolov5;
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
