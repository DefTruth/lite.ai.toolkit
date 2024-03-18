//
// Created by DefTruth on 2021/7/24.
//

#include "lite/lite.h"

static void test_default()
{
  std::string onnx_path = "../../../examples/hub/onnx/cv/face-emotion-recognition-mobilenet_7.onnx";
  std::string test_img_path = "../../../examples/lite/resources/test_lite_emotion.jpg";
  std::string save_img_path = "../../../examples/logs/test_lite_mobile_emotion7.jpg";

  lite::cv::face::attr::MobileEmotion7 *mobile_emotion7 =
      new lite::cv::face::attr::MobileEmotion7(onnx_path);

  lite::types::Emotions emotions;
  cv::Mat img_bgr = cv::imread(test_img_path);
  mobile_emotion7->detect(img_bgr, emotions);

  lite::utils::draw_emotion_inplace(img_bgr, emotions);

  cv::imwrite(save_img_path, img_bgr);

  std::cout << "Default Version Detected Emotion: " << emotions.text
            << ": " << emotions.score << std::endl;

  delete mobile_emotion7;

}

static void test_onnxruntime()
{
#ifdef ENABLE_ONNXRUNTIME
  std::string onnx_path = "../../../examples/hub/onnx/cv/face-emotion-recognition-mobilenet_7.onnx";
  std::string test_img_path = "../../../examples/lite/resources/test_lite_emotion.jpg";
  std::string save_img_path = "../../../examples/logs/test_lite_mobile_emotion7.jpg";

  lite::onnxruntime::cv::face::attr::MobileEmotion7 *mobile_emotion7 =
      new lite::onnxruntime::cv::face::attr::MobileEmotion7(onnx_path);

  lite::types::Emotions emotions;
  cv::Mat img_bgr = cv::imread(test_img_path);
  mobile_emotion7->detect(img_bgr, emotions);

  lite::utils::draw_emotion_inplace(img_bgr, emotions);

  cv::imwrite(save_img_path, img_bgr);

  std::cout << "ONNXRuntime Version Detected Emotion: " << emotions.text
            << ": " << emotions.score << std::endl;

  delete mobile_emotion7;
#endif
}

static void test_mnn()
{
#ifdef ENABLE_MNN
  std::string mnn_path = "../../../examples/hub/mnn/cv/face-emotion-recognition-mobilenet_7.mnn";
  std::string test_img_path = "../../../examples/lite/resources/test_lite_emotion.jpg";
  std::string save_img_path = "../../../examples/logs/test_lite_mobile_emotion7_mnn.jpg";

  lite::mnn::cv::face::attr::MobileEmotion7 *mobile_emotion7 =
      new lite::mnn::cv::face::attr::MobileEmotion7(mnn_path);

  lite::types::Emotions emotions;
  cv::Mat img_bgr = cv::imread(test_img_path);
  mobile_emotion7->detect(img_bgr, emotions);

  lite::utils::draw_emotion_inplace(img_bgr, emotions);

  cv::imwrite(save_img_path, img_bgr);

  std::cout << "MNN Version Detected Emotion: " << emotions.text
            << ": " << emotions.score << std::endl;

  delete mobile_emotion7;
#endif
}

static void test_ncnn()
{
#ifdef ENABLE_NCNN
  std::string param_path = "../../../examples/hub/ncnn/cv/face-emotion-recognition-mobilenet_7.opt.param";
  std::string bin_path = "../../../examples/hub/ncnn/cv/face-emotion-recognition-mobilenet_7.opt.bin";
  std::string test_img_path = "../../../examples/lite/resources/test_lite_emotion.jpg";
  std::string save_img_path = "../../../examples/logs/test_lite_mobile_emotion7_ncnn.jpg";

  lite::ncnn::cv::face::attr::MobileEmotion7 *mobile_emotion7 =
      new lite::ncnn::cv::face::attr::MobileEmotion7(param_path, bin_path);

  lite::types::Emotions emotions;
  cv::Mat img_bgr = cv::imread(test_img_path);
  mobile_emotion7->detect(img_bgr, emotions);

  lite::utils::draw_emotion_inplace(img_bgr, emotions);

  cv::imwrite(save_img_path, img_bgr);

  std::cout << "NCNN Version Detected Emotion: " << emotions.text
            << ": " << emotions.score << std::endl;

  delete mobile_emotion7;
#endif
}

static void test_tnn()
{
#ifdef ENABLE_TNN
  std::string proto_path = "../../../examples/hub/tnn/cv/face-emotion-recognition-mobilenet_7.opt.tnnproto";
  std::string model_path = "../../../examples/hub/tnn/cv/face-emotion-recognition-mobilenet_7.opt.tnnmodel";
  std::string test_img_path = "../../../examples/lite/resources/test_lite_emotion.jpg";
  std::string save_img_path = "../../../examples/logs/test_lite_mobile_emotion7_tnn.jpg";

  lite::tnn::cv::face::attr::MobileEmotion7 *mobile_emotion7 =
      new lite::tnn::cv::face::attr::MobileEmotion7(proto_path, model_path);

  lite::types::Emotions emotions;
  cv::Mat img_bgr = cv::imread(test_img_path);
  mobile_emotion7->detect(img_bgr, emotions);

  lite::utils::draw_emotion_inplace(img_bgr, emotions);

  cv::imwrite(save_img_path, img_bgr);

  std::cout << "TNN Version Detected Emotion: " << emotions.text
            << ": " << emotions.score << std::endl;

  delete mobile_emotion7;
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
