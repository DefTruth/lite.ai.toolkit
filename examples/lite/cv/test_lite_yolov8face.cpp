//
// Created by ai-test1 on 24-7-8.
//

#include "lite/lite.h"

static void test_default()
{
    std::string onnx_path = "../../../examples/hub/onnx/cv/yoloface_8n.onnx";
    std::string test_img_path = "../../../examples/lite/resources/test_lite_face_detector_2.jpg";
    std::string save_img_path = "../../../examples/logs/test_lite_yolov8face.jpg";

    // 1. Test Default Engine ONNXRuntime
    lite::cv::face::detect::YOLOV8Face *yolov8_face = new lite::cv::face::detect::YOLOV8Face(onnx_path);

    std::vector<lite::types::Boxf> detected_boxes;

    cv::Mat img_bgr = cv::imread(test_img_path);

    yolov8_face->detect(img_bgr, detected_boxes);
    lite::utils::draw_boxes_inplace(img_bgr, detected_boxes);

    cv::imwrite(save_img_path, img_bgr);

    std::cout<<"face detect done!"<<std::endl;

    delete yolov8_face;

}

static void test_tensorrt()
{
#ifdef ENABLE_TENSORRT
    std::string engine_path = "../../../examples/hub/trt/cv/yoloface_fp16.engine";
    std::string test_img_path = "../../../examples/lite/resources/test_lite_yolov5_1.jpg";
    std::string save_img_path = "../../../examples/logs/test_lite_yolov8face_trtbbbb.jpg";

    lite::trt::cv::face::detection::YOLOV8Face *yolov8_face  = new lite::trt::cv::face::detection::YOLOV8Face(engine_path);

    cv::Mat test_image = cv::imread(test_img_path);

    std::vector<lite::types::Boxf> detected_boxes;

    yolov8_face->detect(test_image,detected_boxes,0.5f,0.4f);

    std::cout<<"trt face detect done!"<<std::endl;
    lite::utils::draw_boxes_inplace(test_image, detected_boxes);
    cv::imwrite(save_img_path, test_image);

    delete yolov8_face;
#endif
}



int main(__unused int argc, __unused char *argv[])
{
    test_default();
    test_tensorrt();
    return 0;
}
