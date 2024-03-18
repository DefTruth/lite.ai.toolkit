//
// Created by DefTruth on 2021/6/26.
//

#include "lite/lite.h"

static void test_default()
{
  std::string onnx_path = "../../../examples/hub/onnx/cv/ultraface-rfb-640.onnx";
  std::string test_img_path = "../../../examples/lite/resources/test_lite_ultraface.jpg";
  std::string save_img_path = "../../../examples/logs/test_lite_ultraface.jpg";

  lite::cv::face::detect::UltraFace *ultraface = new lite::cv::face::detect::UltraFace(onnx_path);

  std::vector<lite::types::Boxf> detected_boxes;
  cv::Mat img_bgr = cv::imread(test_img_path);
  ultraface->detect(img_bgr, detected_boxes);

  lite::utils::draw_boxes_inplace(img_bgr, detected_boxes);

  cv::imwrite(save_img_path, img_bgr);

  std::cout << "Default Version Done! Detected Face Num: " << detected_boxes.size() << std::endl;

  delete ultraface;
}

static void test_onnxruntime()
{
#ifdef ENABLE_ONNXRUNTIME
  std::string onnx_path = "../../../examples/hub/onnx/cv/ultraface-rfb-640.onnx";
  std::string test_img_path = "../../../examples/lite/resources/test_lite_ultraface.jpg";
  std::string save_img_path = "../../../examples/logs/test_ultraface_onnx.jpg";

  lite::onnxruntime::cv::face::detect::UltraFace *ultraface =
      new lite::onnxruntime::cv::face::detect::UltraFace(onnx_path);

  std::vector<lite::types::Boxf> detected_boxes;
  cv::Mat img_bgr = cv::imread(test_img_path);
  ultraface->detect(img_bgr, detected_boxes);

  lite::utils::draw_boxes_inplace(img_bgr, detected_boxes);

  cv::imwrite(save_img_path, img_bgr);

  std::cout << "ONNXRuntime Version Done! Detected Face Num: " << detected_boxes.size() << std::endl;

  delete ultraface;
#endif
}

static void test_mnn()
{
#ifdef ENABLE_MNN
  std::string mnn_path = "../../../examples/hub/mnn/cv/ultraface-rfb-640.mnn";
  std::string test_img_path = "../../../examples/lite/resources/test_lite_ultraface.jpg";
  std::string save_img_path = "../../../examples/logs/test_ultraface_mnn.jpg";

  lite::mnn::cv::face::detect::UltraFace *ultraface =
      new lite::mnn::cv::face::detect::UltraFace(mnn_path);

  std::vector<lite::types::Boxf> detected_boxes;
  cv::Mat img_bgr = cv::imread(test_img_path);
  ultraface->detect(img_bgr, detected_boxes);

  lite::utils::draw_boxes_inplace(img_bgr, detected_boxes);

  cv::imwrite(save_img_path, img_bgr);

  std::cout << "MNN Version Done! Detected Face Num: " << detected_boxes.size() << std::endl;

  delete ultraface;
#endif
}

static void test_ncnn()
{
#ifdef ENABLE_NCNN
  std::string param_path = "../../../examples/hub/ncnn/cv/ultraface-rfb-320.param";
  std::string bin_path = "../../../examples/hub/ncnn/cv/ultraface-rfb-320.bin";
  std::string test_img_path = "../../../examples/lite/resources/test_lite_ultraface.jpg";
  std::string save_img_path = "../../../examples/logs/test_ultraface_ncnn.jpg";

  lite::ncnn::cv::face::detect::UltraFace *ultraface =
      new lite::ncnn::cv::face::detect::UltraFace(param_path, bin_path, 1, 320, 240);

  std::vector<lite::types::Boxf> detected_boxes;
  cv::Mat img_bgr = cv::imread(test_img_path);
  ultraface->detect(img_bgr, detected_boxes);

  lite::utils::draw_boxes_inplace(img_bgr, detected_boxes);

  cv::imwrite(save_img_path, img_bgr);

  std::cout << "NCNN Version Done! Detected Face Num: " << detected_boxes.size() << std::endl;

  delete ultraface;
#endif
}

static void test_tnn()
{
#ifdef ENABLE_TNN
  std::string proto_path = "../../../examples/hub/tnn/cv/ultraface-rfb-640.opt.tnnproto";
  std::string model_path = "../../../examples/hub/tnn/cv/ultraface-rfb-640.opt.tnnmodel";
  std::string test_img_path = "../../../examples/lite/resources/test_lite_ultraface.jpg";
  std::string save_img_path = "../../../examples/logs/test_ultraface_tnn.jpg";

  lite::tnn::cv::face::detect::UltraFace *ultraface =
      new lite::tnn::cv::face::detect::UltraFace(proto_path, model_path);

  std::vector<lite::types::Boxf> detected_boxes;
  cv::Mat img_bgr = cv::imread(test_img_path);
  ultraface->detect(img_bgr, detected_boxes);

  lite::utils::draw_boxes_inplace(img_bgr, detected_boxes);

  cv::imwrite(save_img_path, img_bgr);

  std::cout << "TNN Version Done! Detected Face Num: " << detected_boxes.size() << std::endl;

  delete ultraface;
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
