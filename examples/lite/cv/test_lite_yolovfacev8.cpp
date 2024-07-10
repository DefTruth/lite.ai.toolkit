//
// Created by ai-test1 on 24-7-8.
//

#include "lite/lite.h"
#include "lite/ort/cv/yolofacev8.h"

// 检测并绘制检测框
void draw_bboxes(cv::Mat& image, const  std::vector<lite::types::BoundingBoxType<float,float>>& bboxes) {
    for (const auto& bbox : bboxes) {
        cv::rectangle(image, cv::Point(bbox.x1, bbox.y1), cv::Point(bbox.x2, bbox.y2), cv::Scalar(0, 255, 0), 2);
    }
}

static void test_default()
{
    std::string onnx_path = "../../hub/onnx/cv/yoloface_8n.onnx";
    std::string test_img_path = "../resources/test_lite_face_detector_2.jpg";
    std::string save_img_path = "../Lite-Face-Detect-11111.jpg";

    // 1. Test Default Engine ONNXRuntime
    lite::cv::face::detect::YOLOV8Face *yolov8Face = new lite::cv::face::detect::YOLOV8Face(onnx_path);

    std::vector<lite::types::BoundingBoxType<float,float>> detect_boxs;

    cv::Mat img_bgr = cv::imread(test_img_path);

    yolov8Face->detect(img_bgr, detect_boxs);

    draw_bboxes(img_bgr,detect_boxs);

    cv::imwrite(save_img_path,img_bgr);

    std::cout<<"face detect done!"<<std::endl;

    delete yolov8Face;

}

int main(__unused int argc, __unused char *argv[])
{
    test_default();
    return 0;
}