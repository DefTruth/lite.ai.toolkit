//
// Created by DefTruth on 2022/6/12.
//

#include "lite/lite.h"

static void test_default()
{
  std::string head_seg_onnx_path = "../../../examples/hub/onnx/cv/minivision_head_seg.onnx";
  std::string cartoon_onnx_path = "../../../examples/hub/onnx/cv/minivision_female_photo2cartoon.onnx";
  std::string test_img_path = "../../../examples/lite/resources/test_lite_head_seg.png";
  std::string save_mask_path = "../../../examples/logs/test_lite_female_photo2cartoon_seg.jpg";
  std::string save_cartoon_path = "../../../examples/logs/test_lite_female_photo2cartoon_cartoon.jpg";

  lite::cv::segmentation::HeadSeg *head_seg =
      new lite::cv::segmentation::HeadSeg(head_seg_onnx_path, 4); // 4 threads
  lite::cv::style::FemalePhoto2Cartoon *female_photo2cartoon =
      new lite::cv::style::FemalePhoto2Cartoon(cartoon_onnx_path, 4); // 4 threads

  lite::types::HeadSegContent head_seg_content;
  cv::Mat img_bgr = cv::imread(test_img_path);
  head_seg->detect(img_bgr, head_seg_content);

  if (head_seg_content.flag && !head_seg_content.mask.empty())
  {
    cv::imwrite(save_mask_path, head_seg_content.mask * 255.f);
    std::cout << "Default Version HeadSeg Done!" << std::endl;
    // Female Photo2Cartoon Style Transfer
    lite::types::FemalePhoto2CartoonContent female_cartoon_content;
    female_photo2cartoon->detect(img_bgr, head_seg_content.mask, female_cartoon_content);

    if (female_cartoon_content.flag && !female_cartoon_content.cartoon.empty())
    {
      cv::imwrite(save_cartoon_path, female_cartoon_content.cartoon);
      std::cout << "Default Version FemalePhoto2Cartoon Done!" << std::endl;
    }
  }

  delete head_seg;
  delete female_photo2cartoon;
}

static void test_onnxruntime()
{
#ifdef ENABLE_ONNXRUNTIME
  std::string head_seg_onnx_path = "../../../examples/hub/onnx/cv/minivision_head_seg.onnx";
  std::string cartoon_onnx_path = "../../../examples/hub/onnx/cv/minivision_female_photo2cartoon.onnx";
  std::string test_img_path = "../../../examples/lite/resources/test_lite_female_photo2cartoon.jpg";
  std::string save_mask_path = "../../../examples/logs/test_lite_female_photo2cartoon_seg_1_onnx.jpg";
  std::string save_cartoon_path = "../../../examples/logs/test_lite_female_photo2cartoon_cartoon_1_onnx.jpg";

  lite::onnxruntime::cv::segmentation::HeadSeg *head_seg =
      new lite::onnxruntime::cv::segmentation::HeadSeg(head_seg_onnx_path, 4); // 4 threads
  lite::onnxruntime::cv::style::FemalePhoto2Cartoon *female_photo2cartoon =
      new lite::onnxruntime::cv::style::FemalePhoto2Cartoon(cartoon_onnx_path, 4); // 4 threads

  lite::types::HeadSegContent head_seg_content;
  cv::Mat img_bgr = cv::imread(test_img_path);
  head_seg->detect(img_bgr, head_seg_content);

  if (head_seg_content.flag && !head_seg_content.mask.empty())
  {
    cv::imwrite(save_mask_path, head_seg_content.mask * 255.f);
    std::cout << "ONNXRuntime Version HeadSeg Done!" << std::endl;
    // Female Photo2Cartoon Style Transfer
    lite::types::FemalePhoto2CartoonContent female_cartoon_content;
    female_photo2cartoon->detect(img_bgr, head_seg_content.mask, female_cartoon_content);

    if (female_cartoon_content.flag && !female_cartoon_content.cartoon.empty())
    {
      cv::imwrite(save_cartoon_path, female_cartoon_content.cartoon);
      std::cout << "ONNXRuntime Version FemalePhoto2Cartoon Done!" << std::endl;
    }
  }

  delete head_seg;
  delete female_photo2cartoon;
#endif
}

static void test_mnn()
{
#ifdef ENABLE_MNN
  std::string head_seg_mnn_path = "../../../examples/hub/mnn/cv/minivision_head_seg.mnn";
  std::string cartoon_mnn_path = "../../../examples/hub/mnn/cv/minivision_female_photo2cartoon.mnn";
  std::string test_img_path = "../../../examples/lite/resources/test_lite_female_photo2cartoon.jpg";
  std::string save_mask_path = "../../../examples/logs/test_lite_female_photo2cartoon_seg_1_mnn.jpg";
  std::string save_cartoon_path = "../../../examples/logs/test_lite_female_photo2cartoon_cartoon_1_mnn.jpg";

  lite::mnn::cv::segmentation::HeadSeg *head_seg =
      new lite::mnn::cv::segmentation::HeadSeg(head_seg_mnn_path, 4); // 4 threads
  lite::mnn::cv::style::FemalePhoto2Cartoon *female_photo2cartoon =
      new lite::mnn::cv::style::FemalePhoto2Cartoon(cartoon_mnn_path, 4); // 4 threads

  lite::types::HeadSegContent head_seg_content;
  cv::Mat img_bgr = cv::imread(test_img_path);
  head_seg->detect(img_bgr, head_seg_content);

  if (head_seg_content.flag && !head_seg_content.mask.empty())
  {
    cv::imwrite(save_mask_path, head_seg_content.mask * 255.f);
    std::cout << "MNN Version HeadSeg Done!" << std::endl;
    // Female Photo2Cartoon Style Transfer
    lite::types::FemalePhoto2CartoonContent female_cartoon_content;
    female_photo2cartoon->detect(img_bgr, head_seg_content.mask, female_cartoon_content);

    if (female_cartoon_content.flag && !female_cartoon_content.cartoon.empty())
    {
      cv::imwrite(save_cartoon_path, female_cartoon_content.cartoon);
      std::cout << "MNN Version FemalePhoto2Cartoon Done!" << std::endl;
    }
  }

  delete head_seg;
  delete female_photo2cartoon;
#endif
}

static void test_ncnn()
{
#ifdef ENABLE_NCNN
  // WARN: TEST FAILED !!!
  std::string head_seg_onnx_path = "../../../examples/hub/onnx/cv/minivision_head_seg.onnx"; // helper
  std::string cartoon_param_path = "../../../examples/hub/ncnn/cv/minivision_female_photo2cartoon.opt.param";
  std::string cartoon_bin_path = "../../../examples/hub/ncnn/cv/minivision_female_photo2cartoon.opt.bin";
  std::string test_img_path = "../../../examples/lite/resources/test_lite_female_photo2cartoon.jpg";
  std::string save_mask_path = "../../../examples/logs/test_lite_female_photo2cartoon_seg_1_ncnn.jpg";
  std::string save_cartoon_path = "../../../examples/logs/test_lite_female_photo2cartoon_cartoon_1_ncnn.jpg";

  lite::cv::segmentation::HeadSeg *head_seg =
      new lite::cv::segmentation::HeadSeg(head_seg_onnx_path, 4); // 4 threads
  lite::ncnn::cv::style::FemalePhoto2Cartoon *female_photo2cartoon =
      new lite::ncnn::cv::style::FemalePhoto2Cartoon(
          cartoon_param_path, cartoon_bin_path, 4); // 4 threads

  lite::types::HeadSegContent head_seg_content;
  cv::Mat img_bgr = cv::imread(test_img_path);
  head_seg->detect(img_bgr, head_seg_content);

  if (head_seg_content.flag && !head_seg_content.mask.empty())
  {
    cv::imwrite(save_mask_path, head_seg_content.mask * 255.f);
    std::cout << "NCNN Version HeadSeg Done!" << std::endl;
    // Female Photo2Cartoon Style Transfer
    lite::types::FemalePhoto2CartoonContent female_cartoon_content;
    female_photo2cartoon->detect(img_bgr, head_seg_content.mask, female_cartoon_content);

    if (female_cartoon_content.flag && !female_cartoon_content.cartoon.empty())
    {
      cv::imwrite(save_cartoon_path, female_cartoon_content.cartoon);
      std::cout << "NCNN Version FemalePhoto2Cartoon Done!" << std::endl;
    }
  }

  delete head_seg;
  delete female_photo2cartoon;
#endif
}

static void test_tnn()
{
#ifdef ENABLE_TNN
  std::string head_seg_onnx_path = "../../../examples/hub/onnx/cv/minivision_head_seg.onnx"; // helper
  std::string cartoon_proto_path = "../../../examples/hub/tnn/cv/minivision_female_photo2cartoon.opt.tnnproto";
  std::string cartoon_model_path = "../../../examples/hub/tnn/cv/minivision_female_photo2cartoon.opt.tnnmodel";
  std::string test_img_path = "../../../examples/lite/resources/test_lite_female_photo2cartoon.jpg";
  std::string save_mask_path = "../../../examples/logs/test_lite_female_photo2cartoon_seg_1_tnn.jpg";
  std::string save_cartoon_path = "../../../examples/logs/test_lite_female_photo2cartoon_cartoon_1_tnn.jpg";

  lite::cv::segmentation::HeadSeg *head_seg =
      new lite::cv::segmentation::HeadSeg(head_seg_onnx_path, 4); // 4 threads
  lite::tnn::cv::style::FemalePhoto2Cartoon *female_photo2cartoon =
      new lite::tnn::cv::style::FemalePhoto2Cartoon(
          cartoon_proto_path, cartoon_model_path, 4); // 4 threads

  lite::types::HeadSegContent head_seg_content;
  cv::Mat img_bgr = cv::imread(test_img_path);
  head_seg->detect(img_bgr, head_seg_content);

  if (head_seg_content.flag && !head_seg_content.mask.empty())
  {
    cv::imwrite(save_mask_path, head_seg_content.mask * 255.f);
    std::cout << "TNN Version HeadSeg Done!" << std::endl;
    // Female Photo2Cartoon Style Transfer
    lite::types::FemalePhoto2CartoonContent female_cartoon_content;
    female_photo2cartoon->detect(img_bgr, head_seg_content.mask, female_cartoon_content);

    if (female_cartoon_content.flag && !female_cartoon_content.cartoon.empty())
    {
      cv::imwrite(save_cartoon_path, female_cartoon_content.cartoon);
      std::cout << "TNN Version FemalePhoto2Cartoon Done!" << std::endl;
    }
  }

  delete head_seg;
  delete female_photo2cartoon;
#endif
}

static void test_lite()
{
  test_default();
  test_onnxruntime();
  test_mnn();
  // test_ncnn();
  test_tnn();
}

int main(__unused int argc, __unused char *argv[])
{
  test_lite();
  return 0;
}
