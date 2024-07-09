//
// Created by ai-test1 on 24-7-8.
//

#include "yolofacev8.h"
using namespace cv;
using namespace ortcv;
using namespace std;
#include "lite/ort/core/ort_utils.h" // 引入onnxruntime特定的自定义功能函数，依赖于推理引擎
#include "lite/utils.h" // 引入全局定义的功能函数，不依赖推理引擎，如NMS


float YoloFaceV8::GetIoU(const lite::types::Bbox box1, const lite::types::Bbox box2) {
    // 获取IOU参数
    float x1 = max(box1.xmin, box2.xmin);
    float y1 = max(box1.ymin, box2.ymin);
    float x2 = min(box1.xmax, box2.xmax);
    float y2 = min(box1.ymax, box2.ymax);
    float w = max(0.f, x2 - x1);
    float h = max(0.f, y2 - y1);
    float over_area = w * h;
    if (over_area == 0)
        return 0.0;
    float union_area = (box1.xmax - box1.xmin) * (box1.ymax - box1.ymin) + (box2.xmax - box2.xmin) * (box2.ymax - box2.ymin) - over_area;
    return over_area / union_area;
}

// NMS参数
std::vector<int> YoloFaceV8::nms(std::vector<lite::types::Bbox> boxes, std::vector<float> confidences, const float nms_thresh) {
    sort(confidences.begin(), confidences.end(), [&confidences](size_t index_1, size_t index_2)
    { return confidences[index_1] > confidences[index_2]; });
    const int num_box = confidences.size();
    vector<bool> isSuppressed(num_box, false);
    for (int i = 0; i < num_box; ++i)
    {
        if (isSuppressed[i])
        {
            continue;
        }
        for (int j = i + 1; j < num_box; ++j)
        {
            if (isSuppressed[j])
            {
                continue;
            }

            float ovr = GetIoU(boxes[i], boxes[j]);
            if (ovr > nms_thresh)
            {
                isSuppressed[j] = true;
            }
        }
    }

    vector<int> keep_inds;
    for (int i = 0; i < isSuppressed.size(); i++)
    {
        if (!isSuppressed[i])
        {
            keep_inds.emplace_back(i);
        }
    }
    return keep_inds;
}


// 归一化
cv::Mat YoloFaceV8::normalize(cv::Mat srcimg) {
    const int height = srcimg.rows;
    const int width = srcimg.cols;
    Mat temp_image = srcimg.clone();
    int input_height = 640;
    int input_width = 640;

    if (height > input_height || width > input_width)
    {
        const float scale = std::min((float)input_height / height, (float)input_width / width);
        Size new_size = Size(int(width * scale), int(height * scale));
        resize(srcimg, temp_image, new_size);
    }

    ratio_height = (float)height / temp_image.rows;
    ratio_width = (float)width / temp_image.cols;

    Mat input_img;
    copyMakeBorder(temp_image, input_img, 0, input_height - temp_image.rows, 0, input_width - temp_image.cols, BORDER_CONSTANT, 0);

    vector<cv::Mat> bgrChannels(3);
    split(input_img, bgrChannels);
    for (int c = 0; c < 3; c++)
    {
        bgrChannels[c].convertTo(bgrChannels[c], CV_32FC1, 1 / 128.0, -127.5 / 128.0);
    }
    cv::Mat normalized_image;
    merge(bgrChannels,normalized_image);
    return normalized_image;

}


// 返回Ort的Value放在最后写
Ort::Value YoloFaceV8::transform(const cv::Mat &mat_rs) {
    // 得到归一化之后的Mat
//    cv::Mat resize_mat = YoloFaceV8::normalize(mat_rs);

    return ortcv::utils::transform::create_tensor(
            mat_rs, input_node_dims, memory_info_handler,
            input_values_handler, ortcv::utils::transform::CHW);
}


//void YoloFaceV8::generate_box(std::vector<Ort::Value> &ort_outputs,std::vector<Bbox> &boxes)
void YoloFaceV8::generate_box(std::vector<Ort::Value> &ort_outputs,std::vector<lite::types::Bbox> &boxes)
{
    float *pdata = ort_outputs[0].GetTensorMutableData<float>(); /// 形状是(1, 20, 8400),不考虑第0维batchsize，每一列的长度20,前4个元素是检测框坐标(cx,cy,w,h)，第4个元素是置信度，剩下的15个元素是5个关键点坐标x,y和置信度
    const int num_box = ort_outputs[0].GetTensorTypeAndShapeInfo().GetShape()[2];
    vector<lite::types::Bbox> bounding_box_raw;
    vector<float> score_raw;
    for (int i = 0; i < num_box; i++)
    {
        const float score = pdata[4 * num_box + i];
        if (score > conf_threshold)
        {
            float xmin = (pdata[i] - 0.5 * pdata[2 * num_box + i]) * ratio_width;            ///(cx,cy,w,h)转到(x,y,w,h)并还原到原图
            float ymin = (pdata[num_box + i] - 0.5 * pdata[3 * num_box + i]) * ratio_height; ///(cx,cy,w,h)转到(x,y,w,h)并还原到原图
            float xmax = (pdata[i] + 0.5 * pdata[2 * num_box + i]) * ratio_width;            ///(cx,cy,w,h)转到(x,y,w,h)并还原到原图
            float ymax = (pdata[num_box + i] + 0.5 * pdata[3 * num_box + i]) * ratio_height; ///(cx,cy,w,h)转到(x,y,w,h)并还原到原图
            ////坐标的越界检查保护，可以添加一下
            bounding_box_raw.emplace_back(lite::types::Bbox{xmin, ymin, xmax, ymax});
            score_raw.emplace_back(score);

        }
    }
    vector<int> keep_inds = nms(bounding_box_raw, score_raw, iou_threshold);
    const int keep_num = keep_inds.size();
    boxes.clear();
    boxes.resize(keep_num);
    for (int i = 0; i < keep_num; i++)
    {
        const int ind = keep_inds[i];
        boxes[i] = bounding_box_raw[ind];
    }

}



// 函数：保存 Ort::Value 数据到文本文件
void save_tensor_to_file(const Ort::Value& tensor, const std::string& filename) {
    // 获取张量的类型和形状信息
    auto type_and_shape_info = tensor.GetTensorTypeAndShapeInfo();
    std::vector<int64_t> shape = type_and_shape_info.GetShape();
    size_t element_count = type_and_shape_info.GetElementCount();

    // 检查数据类型是否为浮点数
    ONNXTensorElementDataType type = type_and_shape_info.GetElementType();
    if (type != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT) {
        std::cerr << "Unsupported tensor data type. Only float tensors are supported." << std::endl;
        return;
    }

    // 使用 GetTensorData<float>() 获取数据指针
    const float* pdata = tensor.GetTensorData<float>();

    // 将数据写入文件
    std::ofstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Could not open file for writing: " << filename << std::endl;
        return;
    }

    for (size_t i = 0; i < element_count; ++i) {
        file << pdata[i] << "\n";
    }
    file.close();
}




void YoloFaceV8::detect(const cv::Mat &mat,std::vector<lite::types::Bbox> &boxes) {
    // 确认基本信息

    // 传入进来的是待检测的图片 真好在这里把缩放值计算一下
    if (mat.empty()) return;

    cv::Mat mat_rs = YoloFaceV8::normalize(mat);

    // 1. make input tensor
    Ort::Value input_tensor = this->transform(mat_rs);

    // 打印输出对比
    // save_tensor_to_file(input_tensor,"./lite.ai.toolkit/TestExamples/input-2012.txt");

    Ort::RunOptions runOptions;

    // 2. inference scores & boxes.
    auto output_tensors = ort_session->Run(
            runOptions, input_node_names.data(),
            &input_tensor, 1, output_node_names.data(), num_outputs
    );

    // save_tensor_to_file(output_tensors[0],"/home/ai-test1/wangzijian/lite.ai.toolkit/TestExamples/out.txt");

    this->generate_box(output_tensors,boxes);


}