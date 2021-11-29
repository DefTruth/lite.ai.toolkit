//
// Created by DefTruth on 2021/6/26.
//

#include "lite/lite.h"

static void test_default()
{
  std::string onnx_path = "../../../hub/onnx/cv/hardnet.onnx";
  std::string test_img_path = "../../../examples/lite/resources/test_lite_hardnet.jpg";

  lite::cv::classification::HdrDNet *hardnet = new lite::cv::classification::HdrDNet(onnx_path);

  lite::types::ImageNetContent content;
  cv::Mat img_bgr = cv::imread(test_img_path);
  hardnet->detect(img_bgr, content);

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

  delete hardnet;
}

static void test_onnxruntime()
{
#ifdef ENABLE_ONNXRUNTIME
  std::string onnx_path = "../../../hub/onnx/cv/hardnet.onnx";
  std::string test_img_path = "../../../examples/lite/resources/test_lite_hardnet.jpg";

  lite::onnxruntime::cv::classification::HdrDNet *hardnet =
      new lite::onnxruntime::cv::classification::HdrDNet(onnx_path);

  lite::types::ImageNetContent content;
  cv::Mat img_bgr = cv::imread(test_img_path);
  hardnet->detect(img_bgr, content);

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

  delete hardnet;
#endif
}

static void test_mnn()
{
#ifdef ENABLE_MNN
  std::string mnn_path = "../../../hub/mnn/cv/hardnet.mnn";
  std::string test_img_path = "../../../examples/lite/resources/test_lite_hardnet.jpg";

  lite::mnn::cv::classification::HdrDNet *hardnet =
      new lite::mnn::cv::classification::HdrDNet(mnn_path);

  lite::types::ImageNetContent content;
  cv::Mat img_bgr = cv::imread(test_img_path);
  hardnet->detect(img_bgr, content);

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

  delete hardnet;
#endif
}

static void test_ncnn()
{
#ifdef ENABLE_NCNN
  std::string param_path = "../../../hub/ncnn/cv/hardnet.opt.param";
  std::string bin_path = "../../../hub/ncnn/cv/hardnet.opt.bin";
  std::string test_img_path = "../../../examples/lite/resources/test_lite_hardnet.jpg";

  lite::ncnn::cv::classification::HdrDNet *hardnet =
      new lite::ncnn::cv::classification::HdrDNet(param_path, bin_path);

  lite::types::ImageNetContent content;
  cv::Mat img_bgr = cv::imread(test_img_path);
  hardnet->detect(img_bgr, content);

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

  delete hardnet;

#endif
}

static void test_tnn()
{
#ifdef ENABLE_TNN
  std::string proto_path = "../../../hub/tnn/cv/hardnet.opt.tnnproto";
  std::string model_path = "../../../hub/tnn/cv/hardnet.opt.tnnmodel";
  std::string test_img_path = "../../../examples/lite/resources/test_lite_hardnet.jpg";

  lite::tnn::cv::classification::HdrDNet *hardnet =
      new lite::tnn::cv::classification::HdrDNet(proto_path, model_path);

  lite::types::ImageNetContent content;
  cv::Mat img_bgr = cv::imread(test_img_path);
  hardnet->detect(img_bgr, content);

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

  delete hardnet;
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
