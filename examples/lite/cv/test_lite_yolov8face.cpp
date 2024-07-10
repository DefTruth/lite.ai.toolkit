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

int main(__unused int argc, __unused char *argv[])
{
    test_default();
    return 0;
}
