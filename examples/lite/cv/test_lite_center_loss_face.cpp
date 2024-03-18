//
// Created by DefTruth on 2021/7/14.
//

#include "lite/lite.h"

static void test_default()
{
  std::string onnx_path = "../../../examples/hub/onnx/cv/CenterLossFace_epoch_100.onnx";
  std::string test_img_path0 = "../../../examples/lite/resources/test_lite_faceid_0.png";
  std::string test_img_path1 = "../../../examples/lite/resources/test_lite_faceid_1.png";

  lite::cv::faceid::CenterLossFace *center_loss_face = new lite::cv::faceid::CenterLossFace(onnx_path);

  lite::types::FaceContent face_content0, face_content1;
  cv::Mat img_bgr0 = cv::imread(test_img_path0);
  cv::Mat img_bgr1 = cv::imread(test_img_path1);
  center_loss_face->detect(img_bgr0, face_content0);
  center_loss_face->detect(img_bgr1, face_content1);

  if (face_content0.flag && face_content1.flag)
  {
    float sim = lite::utils::math::cosine_similarity<float>(
        face_content0.embedding, face_content1.embedding);
    std::cout << "Default Version Detected Sim: " << sim << std::endl;
  }

  delete center_loss_face;
}

static void test_onnxruntime()
{
#ifdef ENABLE_ONNXRUNTIME
  std::string onnx_path = "../../../examples/hub/onnx/cv/CenterLossFace_epoch_100.onnx";
  std::string test_img_path0 = "../../../examples/lite/resources/test_lite_faceid_0.png";
  std::string test_img_path1 = "../../../examples/lite/resources/test_lite_faceid_2.png";

  lite::onnxruntime::cv::faceid::CenterLossFace *center_loss_face =
      new lite::onnxruntime::cv::faceid::CenterLossFace(onnx_path);

  lite::types::FaceContent face_content0, face_content1;
  cv::Mat img_bgr0 = cv::imread(test_img_path0);
  cv::Mat img_bgr1 = cv::imread(test_img_path1);
  center_loss_face->detect(img_bgr0, face_content0);
  center_loss_face->detect(img_bgr1, face_content1);

  if (face_content0.flag && face_content1.flag)
  {
    float sim = lite::utils::math::cosine_similarity<float>(
        face_content0.embedding, face_content1.embedding);
    std::cout << "ONNXRuntime Version Detected Sim: " << sim << std::endl;
  }

  delete center_loss_face;
#endif
}

static void test_mnn()
{
#ifdef ENABLE_MNN
  std::string mnn_path = "../../../examples/hub/mnn/cv/CenterLossFace_epoch_100.mnn";
  std::string test_img_path0 = "../../../examples/lite/resources/test_lite_faceid_0.png";
  std::string test_img_path1 = "../../../examples/lite/resources/test_lite_faceid_2.png";

  lite::mnn::cv::faceid::CenterLossFace *center_loss_face =
      new lite::mnn::cv::faceid::CenterLossFace(mnn_path);

  lite::types::FaceContent face_content0, face_content1;
  cv::Mat img_bgr0 = cv::imread(test_img_path0);
  cv::Mat img_bgr1 = cv::imread(test_img_path1);
  center_loss_face->detect(img_bgr0, face_content0);
  center_loss_face->detect(img_bgr1, face_content1);

  if (face_content0.flag && face_content1.flag)
  {
    float sim = lite::utils::math::cosine_similarity<float>(
        face_content0.embedding, face_content1.embedding);
    std::cout << "MNN Version Detected Sim: " << sim << std::endl;
  }

  delete center_loss_face;
#endif
}

static void test_ncnn()
{
#ifdef ENABLE_NCNN
  std::string param_path = "../../../examples/hub/ncnn/cv/CenterLossFace_epoch_100.opt.param";
  std::string bin_path = "../../../examples/hub/ncnn/cv/CenterLossFace_epoch_100.opt.bin";
  std::string test_img_path0 = "../../../examples/lite/resources/test_lite_faceid_0.png";
  std::string test_img_path1 = "../../../examples/lite/resources/test_lite_faceid_2.png";

  lite::ncnn::cv::faceid::CenterLossFace *center_loss_face =
      new lite::ncnn::cv::faceid::CenterLossFace(param_path, bin_path);

  lite::types::FaceContent face_content0, face_content1;
  cv::Mat img_bgr0 = cv::imread(test_img_path0);
  cv::Mat img_bgr1 = cv::imread(test_img_path1);
  center_loss_face->detect(img_bgr0, face_content0);
  center_loss_face->detect(img_bgr1, face_content1);

  if (face_content0.flag && face_content1.flag)
  {
    float sim = lite::utils::math::cosine_similarity<float>(
        face_content0.embedding, face_content1.embedding);
    std::cout << "NCNN Version Detected Sim: " << sim << std::endl;
  }

  delete center_loss_face;
#endif
}

static void test_tnn()
{
#ifdef ENABLE_TNN
  std::string proto_path = "../../../examples/hub/tnn/cv/CenterLossFace_epoch_100.tnnproto";
  std::string model_path = "../../../examples/hub/tnn/cv/CenterLossFace_epoch_100.tnnmodel";
  std::string test_img_path0 = "../../../examples/lite/resources/test_lite_faceid_0.png";
  std::string test_img_path1 = "../../../examples/lite/resources/test_lite_faceid_2.png";

  lite::tnn::cv::faceid::CenterLossFace *center_loss_face =
      new lite::tnn::cv::faceid::CenterLossFace(proto_path, model_path);

  lite::types::FaceContent face_content0, face_content1;
  cv::Mat img_bgr0 = cv::imread(test_img_path0);
  cv::Mat img_bgr1 = cv::imread(test_img_path1);
  center_loss_face->detect(img_bgr0, face_content0);
  center_loss_face->detect(img_bgr1, face_content1);

  if (face_content0.flag && face_content1.flag)
  {
    float sim = lite::utils::math::cosine_similarity<float>(
        face_content0.embedding, face_content1.embedding);
    std::cout << "TNN Version Detected Sim: " << sim << std::endl;
  }

  delete center_loss_face;
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
