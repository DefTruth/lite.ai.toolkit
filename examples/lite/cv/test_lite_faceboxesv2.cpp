//
// Created by DefTruth on 2022/3/19.
//

#include "lite/lite.h"

static void test_default()
{
  std::string onnx_path = "../../../hub/onnx/cv/faceboxesv2-640x640.onnx";
  std::string test_img_path = "../../../examples/lite/resources/test_lite_face_detector.jpg";
  std::string save_img_path = "../../../logs/test_lite_faceboxesv2.jpg";

  lite::cv::face::detect::FaceBoxesV2 *faceboxesv2 = new lite::cv::face::detect::FaceBoxesV2(onnx_path);

  std::vector<lite::types::Boxf> detected_boxes;
  cv::Mat img_bgr = cv::imread(test_img_path);
  faceboxesv2->detect(img_bgr, detected_boxes);

  lite::utils::draw_boxes_inplace(img_bgr, detected_boxes);

  cv::imwrite(save_img_path, img_bgr);

  std::cout << "Default Version Done! Detected Face Num: " << detected_boxes.size() << std::endl;

  delete faceboxesv2;
}

static void test_onnxruntime()
{
#ifdef ENABLE_ONNXRUNTIME
  std::string onnx_path = "../../../hub/onnx/cv/faceboxesv2-640x640.onnx";
  std::string test_img_path = "../../../examples/lite/resources/test_lite_face_detector.jpg";
  std::string save_img_path = "../../../logs/test_lite_faceboxesv2_onnx.jpg";

  lite::onnxruntime::cv::face::detect::FaceBoxesV2 *faceboxesv2 =
      new lite::onnxruntime::cv::face::detect::FaceBoxesV2(onnx_path);

  std::vector<lite::types::Boxf> detected_boxes;
  cv::Mat img_bgr = cv::imread(test_img_path);
  faceboxesv2->detect(img_bgr, detected_boxes);

  lite::utils::draw_boxes_inplace(img_bgr, detected_boxes);

  cv::imwrite(save_img_path, img_bgr);

  std::cout << "ONNXRuntime Version Done! Detected Face Num: " << detected_boxes.size() << std::endl;

  delete faceboxesv2;
#endif
}

static void test_mnn()
{
#ifdef ENABLE_MNN
  std::string mnn_path = "../../../hub/mnn/cv/faceboxesv2-640x640.mnn";
  std::string test_img_path = "../../../examples/lite/resources/test_lite_face_detector.jpg";
  std::string save_img_path = "../../../logs/test_lite_faceboxesv2_mnn.jpg";

  lite::mnn::cv::face::detect::FaceBoxesV2 *faceboxesv2 =
      new lite::mnn::cv::face::detect::FaceBoxesV2(mnn_path);

  std::vector<lite::types::Boxf> detected_boxes;
  cv::Mat img_bgr = cv::imread(test_img_path);
  faceboxesv2->detect(img_bgr, detected_boxes);

  lite::utils::draw_boxes_inplace(img_bgr, detected_boxes);

  cv::imwrite(save_img_path, img_bgr);

  std::cout << "MNN Version Done! Detected Face Num: " << detected_boxes.size() << std::endl;

  delete faceboxesv2;
#endif
}

static void test_ncnn()
{
#ifdef ENABLE_NCNN
  std::string param_path = "../../../hub/ncnn/cv/faceboxesv2-640x640.opt.param";
  std::string bin_path = "../../../hub/ncnn/cv/faceboxesv2-640x640.opt.bin";
  std::string test_img_path = "../../../examples/lite/resources/test_lite_face_detector.jpg";
  std::string save_img_path = "../../../logs/test_lite_faceboxesv2_ncnn.jpg";

  lite::ncnn::cv::face::detect::FaceBoxesV2 *faceboxesv2 =
      new lite::ncnn::cv::face::detect::FaceBoxesV2(param_path, bin_path);

  std::vector<lite::types::Boxf> detected_boxes;
  cv::Mat img_bgr = cv::imread(test_img_path);
  faceboxesv2->detect(img_bgr, detected_boxes);

  lite::utils::draw_boxes_inplace(img_bgr, detected_boxes);

  cv::imwrite(save_img_path, img_bgr);

  std::cout << "NCNN Version Done! Detected Face Num: " << detected_boxes.size() << std::endl;

  delete faceboxesv2;
#endif
}

static void test_tnn()
{
#ifdef ENABLE_TNN
  std::string proto_path = "../../../hub/tnn/cv/faceboxesv2-640x640.opt.tnnproto";
  std::string model_path = "../../../hub/tnn/cv/faceboxesv2-640x640.opt.tnnmodel";
  std::string test_img_path = "../../../examples/lite/resources/test_lite_face_detector.jpg";
  std::string save_img_path = "../../../logs/test_lite_faceboxesv2_tnn.jpg";

  lite::tnn::cv::face::detect::FaceBoxesV2 *faceboxesv2 =
      new lite::tnn::cv::face::detect::FaceBoxesV2(proto_path, model_path);

  std::vector<lite::types::Boxf> detected_boxes;
  cv::Mat img_bgr = cv::imread(test_img_path);
  faceboxesv2->detect(img_bgr, detected_boxes);

  lite::utils::draw_boxes_inplace(img_bgr, detected_boxes);

  cv::imwrite(save_img_path, img_bgr);

  std::cout << "TNN Version Done! Detected Face Num: " << detected_boxes.size() << std::endl;

  delete faceboxesv2;
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