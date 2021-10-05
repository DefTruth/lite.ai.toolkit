//
// Created by DefTruth on 2021/9/14.
//

#include "lite/lite.h"

static void test_default()
{
  std::string onnx_path = "../../../hub/onnx/cv/yolop-640-640.onnx";
  std::string test_img_path = "../../../examples/lite/resources/test_lite_yolop.jpg";
  std::string save_det_path = "../../../logs/test_lite_yolop_det.jpg";
  std::string save_da_path = "../../../logs/test_lite_yolop_da.jpg";
  std::string save_ll_path = "../../../logs/test_lite_yolop_ll.jpg";
  std::string save_merge_path = "../../../logs/test_lite_yolop_merge.jpg";

  lite::cv::detection::YOLOP *yolop = new lite::cv::detection::YOLOP(onnx_path, 16); // 16 threads

  lite::cv::types::SegmentContent da_seg_content;
  lite::cv::types::SegmentContent ll_seg_content;
  std::vector<lite::cv::types::Boxf> detected_boxes;
  cv::Mat img_bgr = cv::imread(test_img_path);
  yolop->detect(img_bgr, detected_boxes, da_seg_content, ll_seg_content);

  if (!detected_boxes.empty() && da_seg_content.flag && ll_seg_content.flag)
  {
    // boxes.
    cv::Mat img_det = img_bgr.clone();
    lite::cv::utils::draw_boxes_inplace(img_det, detected_boxes);
    cv::imwrite(save_det_path, img_det);
    std::cout << "Saved " << save_det_path << " done!" << "\n";
    // da && ll seg
    cv::imwrite(save_da_path, da_seg_content.class_mat);
    cv::imwrite(save_ll_path, ll_seg_content.class_mat);
    std::cout << "Saved " << save_da_path << " done!" << "\n";
    std::cout << "Saved " << save_ll_path << " done!" << "\n";
    // merge
    cv::Mat img_merge = img_bgr.clone();
    cv::Mat color_seg = da_seg_content.color_mat + ll_seg_content.color_mat;

    cv::addWeighted(img_merge, 0.5, color_seg, 0.5, 0., img_merge);
    lite::cv::utils::draw_boxes_inplace(img_merge, detected_boxes);
    cv::imwrite(save_merge_path, img_merge);
    std::cout << "Saved " << save_merge_path << " done!" << "\n";

    // label
    if (!da_seg_content.names_map.empty() && !ll_seg_content.names_map.empty())
    {

      for (auto it = da_seg_content.names_map.begin(); it != da_seg_content.names_map.end(); ++it)
      {
        std::cout << "Default Version Detected Label: "
                  << it->first << " Name: " << it->second << std::endl;
      }

      for (auto it = ll_seg_content.names_map.begin(); it != ll_seg_content.names_map.end(); ++it)
      {
        std::cout << "Default Version Detected Label: "
                  << it->first << " Name: " << it->second << std::endl;
      }

    }
  }

  delete yolop;
}

static void test_onnxruntime()
{
  std::string onnx_path = "../../../hub/onnx/cv/yolop-640-640.onnx";
  std::string test_img_path = "../../../examples/lite/resources/test_lite_yolop.jpg";
  std::string save_det_path = "../../../logs/test_lite_yolop_det_onnx.jpg";
  std::string save_da_path = "../../../logs/test_lite_yolop_da_onnx.jpg";
  std::string save_ll_path = "../../../logs/test_lite_yolop_ll_onnx.jpg";
  std::string save_merge_path = "../../../logs/test_lite_yolop_merge_onnx.jpg";

  lite::onnxruntime::cv::detection::YOLOP *yolop =
      new lite::onnxruntime::cv::detection::YOLOP(onnx_path, 16); // 16 threads

  lite::onnxruntime::cv::types::SegmentContent da_seg_content;
  lite::onnxruntime::cv::types::SegmentContent ll_seg_content;
  std::vector<lite::onnxruntime::cv::types::Boxf> detected_boxes;
  cv::Mat img_bgr = cv::imread(test_img_path);
  yolop->detect(img_bgr, detected_boxes, da_seg_content, ll_seg_content);

  if (!detected_boxes.empty() && da_seg_content.flag && ll_seg_content.flag)
  {
    // boxes.
    cv::Mat img_det = img_bgr.clone();
    lite::onnxruntime::cv::utils::draw_boxes_inplace(img_det, detected_boxes);
    cv::imwrite(save_det_path, img_det);
    std::cout << "Saved " << save_det_path << " done!" << "\n";
    // da && ll seg
    cv::imwrite(save_da_path, da_seg_content.class_mat);
    cv::imwrite(save_ll_path, ll_seg_content.class_mat);
    std::cout << "Saved " << save_da_path << " done!" << "\n";
    std::cout << "Saved " << save_ll_path << " done!" << "\n";
    // merge
    cv::Mat img_merge = img_bgr.clone();
    cv::Mat color_seg = da_seg_content.color_mat + ll_seg_content.color_mat;

    cv::addWeighted(img_merge, 0.5, color_seg, 0.5, 0., img_merge);
    lite::onnxruntime::cv::utils::draw_boxes_inplace(img_merge, detected_boxes);
    cv::imwrite(save_merge_path, img_merge);
    std::cout << "Saved " << save_merge_path << " done!" << "\n";

    // label
    if (!da_seg_content.names_map.empty() && !ll_seg_content.names_map.empty())
    {

      for (auto it = da_seg_content.names_map.begin(); it != da_seg_content.names_map.end(); ++it)
      {
        std::cout << "ONNXRuntime Version Detected Label: "
                  << it->first << " Name: " << it->second << std::endl;
      }

      for (auto it = ll_seg_content.names_map.begin(); it != ll_seg_content.names_map.end(); ++it)
      {
        std::cout << "ONNXRuntime Version Detected Label: "
                  << it->first << " Name: " << it->second << std::endl;
      }

    }
  }

  delete yolop;
}

static void test_mnn()
{
#ifdef ENABLE_MNN
#endif
}

static void test_ncnn()
{
#ifdef ENABLE_NCNN
#endif
}

static void test_tnn()
{
#ifdef ENABLE_TNN
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
