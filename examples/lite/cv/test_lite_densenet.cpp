//
// Created by DefTruth on 2021/6/26.
//

#include "lite/lite.h"

static void test_default()
{
  std::string onnx_path = "../../../examples/hub/onnx/cv/densenet121.onnx";
  std::string test_img_path = "../../../examples/lite/resources/test_lite_densenet.jpg";

  lite::cv::classification::DenseNet *densenet = new lite::cv::classification::DenseNet(onnx_path);

  lite::types::ImageNetContent content;
  cv::Mat img_bgr = cv::imread(test_img_path);
  densenet->detect(img_bgr, content);

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

  delete densenet;
}

static void test_onnxruntime()
{
#ifdef ENABLE_ONNXRUNTIME
  std::string onnx_path = "../../../examples/hub/onnx/cv/densenet121.onnx";
  std::string test_img_path = "../../../examples/lite/resources/test_lite_densenet.jpg";

  lite::onnxruntime::cv::classification::DenseNet *densenet =
      new lite::onnxruntime::cv::classification::DenseNet(onnx_path);

  lite::types::ImageNetContent content;
  cv::Mat img_bgr = cv::imread(test_img_path);
  densenet->detect(img_bgr, content);

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

  delete densenet;
#endif
}

static void test_mnn()
{
#ifdef ENABLE_MNN
  std::string mnn_path = "../../../examples/hub/mnn/cv/densenet121.mnn";
  std::string test_img_path = "../../../examples/lite/resources/test_lite_densenet.jpg";

  lite::mnn::cv::classification::DenseNet *densenet =
      new lite::mnn::cv::classification::DenseNet(mnn_path);

  lite::types::ImageNetContent content;
  cv::Mat img_bgr = cv::imread(test_img_path);
  densenet->detect(img_bgr, content);

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

  delete densenet;
#endif
}

static void test_ncnn()
{
#ifdef ENABLE_NCNN
  std::string param_path = "../../../examples/hub/ncnn/cv/densenet121.opt.param";
  std::string bin_path = "../../../examples/hub/ncnn/cv/densenet121.opt.bin";
  std::string test_img_path = "../../../examples/lite/resources/test_lite_densenet.jpg";

  lite::ncnn::cv::classification::DenseNet *densenet =
      new lite::ncnn::cv::classification::DenseNet(param_path, bin_path);

  lite::types::ImageNetContent content;
  cv::Mat img_bgr = cv::imread(test_img_path);
  densenet->detect(img_bgr, content);

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

  delete densenet;
#endif
}

static void test_tnn()
{
#ifdef ENABLE_TNN
  std::string proto_path = "../../../examples/hub/tnn/cv/densenet121.opt.tnnproto";
  std::string model_path = "../../../examples/hub/tnn/cv/densenet121.opt.tnnmodel";
  std::string test_img_path = "../../../examples/lite/resources/test_lite_densenet.jpg";

  lite::tnn::cv::classification::DenseNet *densenet =
      new lite::tnn::cv::classification::DenseNet(proto_path, model_path);

  lite::types::ImageNetContent content;
  cv::Mat img_bgr = cv::imread(test_img_path);
  densenet->detect(img_bgr, content);

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

  delete densenet;
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
