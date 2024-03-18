//
// Created by DefTruth on 2021/6/26.
//

#include "lite/lite.h"

static void test_default()
{
  std::string onnx_path = "../../../examples/hub/onnx/cv/shufflenet-v2-10.onnx";
  std::string test_img_path = "../../../examples/lite/resources/test_lite_shufflenetv2.jpg";

  lite::cv::classification::ShuffleNetV2 *shufflenetv2 =
      new lite::cv::classification::ShuffleNetV2(onnx_path);

  lite::types::ImageNetContent content;
  cv::Mat img_bgr = cv::imread(test_img_path);
  shufflenetv2->detect(img_bgr, content);

  if (content.flag)
  {
    const unsigned int top_k = content.scores.size();
    if (top_k > 0)
    {
      for (unsigned int i = 0; i < top_k; ++i)
        std::cout << i + 1
                  << ": " << content.labels.at(i)
                  << ": " << content.texts.at(i)
                  << ": " << content.scores.at(i)
                  << std::endl;
    }
    std::cout << "Default Version Done!" << std::endl;
  }

  delete shufflenetv2;
}

static void test_onnxruntime()
{
#ifdef ENABLE_ONNXRUNTIME
  std::string onnx_path = "../../../examples/hub/onnx/cv/shufflenet-v2-10.onnx";
  std::string test_img_path = "../../../examples/lite/resources/test_lite_shufflenetv2.jpg";

  lite::onnxruntime::cv::classification::ShuffleNetV2 *shufflenetv2 =
      new lite::onnxruntime::cv::classification::ShuffleNetV2(onnx_path);

  lite::types::ImageNetContent content;
  cv::Mat img_bgr = cv::imread(test_img_path);
  shufflenetv2->detect(img_bgr, content);

  if (content.flag)
  {
    const unsigned int top_k = content.scores.size();
    if (top_k > 0)
    {
      for (unsigned int i = 0; i < top_k; ++i)
        std::cout << i + 1
                  << ": " << content.labels.at(i)
                  << ": " << content.texts.at(i)
                  << ": " << content.scores.at(i)
                  << std::endl;
    }
    std::cout << "ONNXRuntime Version Done!" << std::endl;
  }

  delete shufflenetv2;
#endif
}

static void test_mnn()
{
#ifdef ENABLE_MNN
  std::string mnn_path = "../../../examples/hub/mnn/cv/shufflenet-v2-10.mnn";
  std::string test_img_path = "../../../examples/lite/resources/test_lite_shufflenetv2.jpg";

  lite::mnn::cv::classification::ShuffleNetV2 *shufflenetv2 =
      new lite::mnn::cv::classification::ShuffleNetV2(mnn_path);

  lite::types::ImageNetContent content;
  cv::Mat img_bgr = cv::imread(test_img_path);
  shufflenetv2->detect(img_bgr, content);

  if (content.flag)
  {
    const unsigned int top_k = content.scores.size();
    if (top_k > 0)
    {
      for (unsigned int i = 0; i < top_k; ++i)
        std::cout << i + 1
                  << ": " << content.labels.at(i)
                  << ": " << content.texts.at(i)
                  << ": " << content.scores.at(i)
                  << std::endl;
    }
    std::cout << "MNN Version Done!" << std::endl;
  }

  delete shufflenetv2;
#endif
}

static void test_ncnn()
{
#ifdef ENABLE_NCNN
  std::string param_path = "../../../examples/hub/ncnn/cv/shufflenet-v2-10.opt.param";
  std::string bin_path = "../../../examples/hub/ncnn/cv/shufflenet-v2-10.opt.bin";
  std::string test_img_path = "../../../examples/lite/resources/test_lite_shufflenetv2.jpg";

  lite::ncnn::cv::classification::ShuffleNetV2 *shufflenetv2 =
      new lite::ncnn::cv::classification::ShuffleNetV2(param_path, bin_path);

  lite::types::ImageNetContent content;
  cv::Mat img_bgr = cv::imread(test_img_path);
  shufflenetv2->detect(img_bgr, content);

  if (content.flag)
  {
    const unsigned int top_k = content.scores.size();
    if (top_k > 0)
    {
      for (unsigned int i = 0; i < top_k; ++i)
        std::cout << i + 1
                  << ": " << content.labels.at(i)
                  << ": " << content.texts.at(i)
                  << ": " << content.scores.at(i)
                  << std::endl;
    }
    std::cout << "NCNN Version Done!" << std::endl;
  }

  delete shufflenetv2;
#endif
}

static void test_tnn()
{
#ifdef ENABLE_TNN
  std::string proto_path = "../../../examples/hub/tnn/cv/shufflenet-v2-10.opt.tnnproto";
  std::string model_path = "../../../examples/hub/tnn/cv/shufflenet-v2-10.opt.tnnmodel";
  std::string test_img_path = "../../../examples/lite/resources/test_lite_shufflenetv2.jpg";

  lite::tnn::cv::classification::ShuffleNetV2 *shufflenetv2 =
      new lite::tnn::cv::classification::ShuffleNetV2(proto_path, model_path);

  lite::types::ImageNetContent content;
  cv::Mat img_bgr = cv::imread(test_img_path);
  shufflenetv2->detect(img_bgr, content);

  if (content.flag)
  {
    const unsigned int top_k = content.scores.size();
    if (top_k > 0)
    {
      for (unsigned int i = 0; i < top_k; ++i)
        std::cout << i + 1
                  << ": " << content.labels.at(i)
                  << ": " << content.texts.at(i)
                  << ": " << content.scores.at(i)
                  << std::endl;
    }
    std::cout << "TNN Version Done!" << std::endl;
  }

  delete shufflenetv2;
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
