//
// Created by DefTruth on 2021/8/1.
//

#include "lite/lite.h"

static void test_default()
{
  std::string onnx_path = "../../../hub/onnx/cv/FaceBoxes.onnx";
  std::string test_img_path = "../../../examples/lite/resources/test_lite_face_detector.jpg";
  std::string save_img_path = "../../../logs/test_lite_faceboxes.jpg";

  lite::cv::face::detect::FaceBoxes *faceboxes = new lite::cv::face::detect::FaceBoxes(onnx_path);

  std::vector<lite::types::Boxf> detected_boxes;
  cv::Mat img_bgr = cv::imread(test_img_path);
  faceboxes->detect(img_bgr, detected_boxes);

  lite::utils::draw_boxes_inplace(img_bgr, detected_boxes);

  cv::imwrite(save_img_path, img_bgr);

  std::cout << "Default Version Done! Detected Face Num: " << detected_boxes.size() << std::endl;

  delete faceboxes;
}

static void test_onnxruntime()
{
#ifdef ENABLE_ONNXRUNTIME
  std::string onnx_path = "../../../hub/onnx/cv/FaceBoxes.onnx";
  std::string test_img_path = "../../../examples/lite/resources/test_lite_face_detector_2.jpg";
  std::string save_img_path = "../../../logs/test_faceboxes_onnx_2.jpg";

  lite::onnxruntime::cv::face::detect::FaceBoxes *faceboxes =
      new lite::onnxruntime::cv::face::detect::FaceBoxes(onnx_path);

  std::vector<lite::types::Boxf> detected_boxes;
  cv::Mat img_bgr = cv::imread(test_img_path);
  faceboxes->detect(img_bgr, detected_boxes);

  lite::utils::draw_boxes_inplace(img_bgr, detected_boxes);

  cv::imwrite(save_img_path, img_bgr);

  std::cout << "ONNXRuntime Version Done! Detected Face Num: " << detected_boxes.size() << std::endl;

  delete faceboxes;
#endif
}

static void test_mnn()
{
#ifdef ENABLE_MNN
  std::string mnn_path = "../../../hub/mnn/cv/FaceBoxes.mnn";
  std::string test_img_path = "../../../examples/lite/resources/test_lite_face_detector_2.jpg";
  std::string save_img_path = "../../../logs/test_faceboxes_mnn_2.jpg";

  lite::mnn::cv::face::detect::FaceBoxes *faceboxes =
      new lite::mnn::cv::face::detect::FaceBoxes(mnn_path);

  std::vector<lite::types::Boxf> detected_boxes;
  cv::Mat img_bgr = cv::imread(test_img_path);
  faceboxes->detect(img_bgr, detected_boxes);

  lite::utils::draw_boxes_inplace(img_bgr, detected_boxes);

  cv::imwrite(save_img_path, img_bgr);

  std::cout << "MNN Version Done! Detected Face Num: " << detected_boxes.size() << std::endl;

  delete faceboxes;
#endif
}

static void test_ncnn()
{
#ifdef ENABLE_NCNN
  std::string param_path = "../../../hub/ncnn/cv/FaceBoxes.opt.param";
  std::string bin_path = "../../../hub/ncnn/cv/FaceBoxes.opt.bin";
  std::string test_img_path = "../../../examples/lite/resources/test_lite_face_detector_2.jpg";
  std::string save_img_path = "../../../logs/test_faceboxes_ncnn_2.jpg";

  lite::ncnn::cv::face::detect::FaceBoxes *faceboxes =
      new lite::ncnn::cv::face::detect::FaceBoxes(param_path, bin_path);

  std::vector<lite::types::Boxf> detected_boxes;
  cv::Mat img_bgr = cv::imread(test_img_path);
  faceboxes->detect(img_bgr, detected_boxes);

  lite::utils::draw_boxes_inplace(img_bgr, detected_boxes);

  cv::imwrite(save_img_path, img_bgr);

  std::cout << "NCNN Version Done! Detected Face Num: " << detected_boxes.size() << std::endl;

  delete faceboxes;
#endif
}

static void test_tnn()
{
#ifdef ENABLE_TNN
  std::string proto_path = "../../../hub/tnn/cv/FaceBoxes.opt.tnnproto";
  std::string model_path = "../../../hub/tnn/cv/FaceBoxes.opt.tnnmodel";
  std::string test_img_path = "../../../examples/lite/resources/test_lite_face_detector_2.jpg";
  std::string save_img_path = "../../../logs/test_faceboxes_tnn_2.jpg";

  lite::tnn::cv::face::detect::FaceBoxes *faceboxes =
      new lite::tnn::cv::face::detect::FaceBoxes(proto_path, model_path);

  std::vector<lite::types::Boxf> detected_boxes;
  cv::Mat img_bgr = cv::imread(test_img_path);
  faceboxes->detect(img_bgr, detected_boxes);

  lite::utils::draw_boxes_inplace(img_bgr, detected_boxes);

  cv::imwrite(save_img_path, img_bgr);

  std::cout << "TNN Version Done! Detected Face Num: " << detected_boxes.size() << std::endl;

  delete faceboxes;
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