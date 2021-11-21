//
// Created by DefTruth on 2021/7/31.
//

#include "lite/lite.h"

static void test_default()
{
  std::string onnx_path = "../../../hub/onnx/cv/Pytorch_RetinaFace_resnet50.onnx";
  std::string test_img_path = "../../../examples/lite/resources/test_lite_face_detector.jpg";
  std::string save_img_path = "../../../logs/test_lite_retinaface.jpg";

  lite::cv::face::detect::RetinaFace *retinaface = new lite::cv::face::detect::RetinaFace(onnx_path);

  std::vector<lite::types::Boxf> detected_boxes;
  cv::Mat img_bgr = cv::imread(test_img_path);
  retinaface->detect(img_bgr, detected_boxes);

  lite::utils::draw_boxes_inplace(img_bgr, detected_boxes);

  cv::imwrite(save_img_path, img_bgr);

  std::cout << "Default Version Done! Detected Face Num: " << detected_boxes.size() << std::endl;

  delete retinaface;
}

static void test_onnxruntime()
{
#ifdef ENABLE_ONNXRUNTIME
  std::string onnx_path = "../../../hub/onnx/cv/Pytorch_RetinaFace_resnet50.onnx";
  std::string test_img_path = "../../../examples/lite/resources/test_lite_face_detector_2.jpg";
  std::string save_img_path = "../../../logs/test_retinaface_onnx_2.jpg";

  lite::onnxruntime::cv::face::detect::RetinaFace *retinaface =
      new lite::onnxruntime::cv::face::detect::RetinaFace(onnx_path);

  std::vector<lite::types::Boxf> detected_boxes;
  cv::Mat img_bgr = cv::imread(test_img_path);
  retinaface->detect(img_bgr, detected_boxes);

  lite::utils::draw_boxes_inplace(img_bgr, detected_boxes);

  cv::imwrite(save_img_path, img_bgr);

  std::cout << "ONNXRuntime Version Done! Detected Face Num: " << detected_boxes.size() << std::endl;

  delete retinaface;
#endif
}

static void test_mnn()
{
#ifdef ENABLE_MNN
  std::string mnn_path = "../../../hub/mnn/cv/Pytorch_RetinaFace_resnet50.mnn";
  std::string test_img_path = "../../../examples/lite/resources/test_lite_face_detector_2.jpg";
  std::string save_img_path = "../../../logs/test_retinaface_mnn_2.jpg";

  lite::mnn::cv::face::detect::RetinaFace *retinaface =
      new lite::mnn::cv::face::detect::RetinaFace(mnn_path);

  std::vector<lite::types::Boxf> detected_boxes;
  cv::Mat img_bgr = cv::imread(test_img_path);
  retinaface->detect(img_bgr, detected_boxes);

  lite::utils::draw_boxes_inplace(img_bgr, detected_boxes);

  cv::imwrite(save_img_path, img_bgr);

  std::cout << "MNN Version Done! Detected Face Num: " << detected_boxes.size() << std::endl;

  delete retinaface;
#endif
}

static void test_ncnn()
{
#ifdef ENABLE_NCNN
  std::string param_path = "../../../hub/ncnn/cv/Pytorch_RetinaFace_mobile0.25-640-640.opt.param";
  std::string bin_path = "../../../hub/ncnn/cv/Pytorch_RetinaFace_mobile0.25-640-640.opt.bin";
  std::string test_img_path = "../../../examples/lite/resources/test_lite_face_detector_2.jpg";
  std::string save_img_path = "../../../logs/test_retinaface_ncnn_2.jpg";

  lite::ncnn::cv::face::detect::RetinaFace *retinaface =
      new lite::ncnn::cv::face::detect::RetinaFace(param_path, bin_path, 1, 640, 640);

  std::vector<lite::types::Boxf> detected_boxes;
  cv::Mat img_bgr = cv::imread(test_img_path);
  retinaface->detect(img_bgr, detected_boxes);

  lite::utils::draw_boxes_inplace(img_bgr, detected_boxes);

  cv::imwrite(save_img_path, img_bgr);

  std::cout << "NCNN Version Done! Detected Face Num: " << detected_boxes.size() << std::endl;

  delete retinaface;
#endif
}

static void test_tnn()
{
#ifdef ENABLE_TNN
  std::string proto_path = "../../../hub/tnn/cv/Pytorch_RetinaFace_mobile0.25-640-640.opt.tnnproto";
  std::string model_path = "../../../hub/tnn/cv/Pytorch_RetinaFace_mobile0.25-640-640.opt.tnnmodel";
  std::string test_img_path = "../../../examples/lite/resources/test_lite_face_detector_2.jpg";
  std::string save_img_path = "../../../logs/test_retinaface_tnn_2.jpg";

  lite::tnn::cv::face::detect::RetinaFace *retinaface =
      new lite::tnn::cv::face::detect::RetinaFace(proto_path, model_path);

  std::vector<lite::types::Boxf> detected_boxes;
  cv::Mat img_bgr = cv::imread(test_img_path);
  retinaface->detect(img_bgr, detected_boxes);

  lite::utils::draw_boxes_inplace(img_bgr, detected_boxes);

  cv::imwrite(save_img_path, img_bgr);

  std::cout << "TNN Version Done! Detected Face Num: " << detected_boxes.size() << std::endl;

  delete retinaface;
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