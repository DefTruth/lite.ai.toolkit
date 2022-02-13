# 如何添加您的模型以及成为贡献者？

## Lite.Ai.ToolKit代码结构介绍

lite.ai.toolkit采用了尽可能解耦的方式来管理代码，基于不同的推理引擎的实现是相互独立的，比如ONNXRuntime版本的YOLOX和MNN版本的YOLOX是相互独立的，他们的代码分别管理在不同的目录下，你可以只编译ONNXRuntime版本的实现。为方便lite.ai.toolkit的用户添加自己的模型，这里简单介绍下lite.ai.toolkit的代码布局。

* lite文件夹  
该文件夹的根目录下包含了所有的主要代码，其结构如下：
```text
# --------------------------- 这部分是管理整理工程的代码 被下游的各个模块依赖 ----------------------------------
├── backend.h          # 宏处理 决定基础的推理引擎 目前必须是ONNXRuntime
├── config.h           # 宏处理
├── config.h.in        # cmake编译时宏处理
├── lite.ai.defs.h     # 宏处理
├── lite.ai.headers.h  # 引入基础依赖库
├── lite.h             # 引入项目的基础对外的模块
├── pipeline           # 暂时没用
├── pipeline.h         # 暂时没用
├── types.h            # 基础类型，很重要，会被下游模块复用，比如实际上ort::types是types的alias
├── utils.cpp          # 基础功能函数实现，很重要，会被下游模块复用
└── utils.h            # 基础功能函数头文件
├── models.h           # 模型整体的命名空间管理，很重要，所有被实现的模型需要在这里被导出

# --------------------------- 以下的各个部分是相互独立的 但是需要依赖上面这个整体的部分 --------------------------
...
├── mnn
│   ├── core     #  MNN基础父类和特定功能的实现，必须要阅读
│   │   ├── mnn_config.h
│   │   ├── mnn_core.h       # MNN模型命名空间管理，实现一个类前，现在这里添加签名
│   │   ├── mnn_defs.h
│   │   ├── mnn_handler.cpp  # 基础父类实现，必须阅读
│   │   ├── mnn_handler.h
│   │   ├── mnn_types.h
│   │   ├── mnn_utils.cpp
│   │   └── mnn_utils.h
│   └── cv      #  各个模型的具体实现，会引用core中实现的父类和功能函数
│       ├── mnn_age_googlenet.cpp
│       ├── mnn_age_googlenet.h
│       ├── mnn_cava_combined_face.cpp
...
├── ncnn
│   ├── core     #  NCNN基础父类和特定功能的实现，必须要阅读
│   │   ├── ncnn_config.h
│   │   ├── ncnn_core.h        # NCNN模型命名空间管理，实现一个类前，现在这里添加签名
│   │   ├── ncnn_custom.cpp
│   │   ├── ncnn_custom.h
│   │   ├── ncnn_defs.h
│   │   ├── ncnn_handler.cpp   # 基础父类实现，必须阅读
│   │   ├── ncnn_handler.h
│   │   ├── ncnn_types.h
│   │   ├── ncnn_utils.cpp
│   │   └── ncnn_utils.h
│   └── cv      #  各个模型的具体实现，会引用core中实现的父类和功能函数
│       ├── ncnn_age_googlenet.cpp
│       ├── ncnn_age_googlenet.h
│       ├── ncnn_cava_combined_face.cpp
│       ├── ncnn_cava_combined_face.h
│       ├── ncnn_cava_ghost_arcface.cpp
...
├── ort
│   ├── core     #  ONNXRuntime基础父类和特定功能的实现，必须要阅读
│   │   ├── ort_config.h
│   │   ├── ort_core.h        # ONNXRuntime模型命名空间管理，实现一个类前，现在这里添加签名
│   │   ├── ort_defs.h
│   │   ├── ort_handler.cpp   # 基础父类实现，必须阅读
│   │   ├── ort_handler.h
│   │   ├── ort_types.h
│   │   ├── ort_utils.cpp
│   │   └── ort_utils.h
│   └── cv      #  各个模型的具体实现，会引用core中实现的父类和功能函数
│       ├── age_googlenet.cpp
│       ├── age_googlenet.h
│       ├── cava_combined_face.cpp
│       ├── cava_combined_face.h
│       ├── cava_ghost_arcface.cpp
│       ├── cava_ghost_arcface.h
...
├── tnn
│   ├── core     #  TNN基础父类和特定功能的实现，必须要阅读
│   │   ├── tnn_config.h
│   │   ├── tnn_core.h        # TNN模型命名空间管理，实现一个类前，现在这里添加签名
│   │   ├── tnn_defs.h
│   │   ├── tnn_handler.cpp   # 基础父类实现，必须阅读
│   │   ├── tnn_handler.h
│   │   ├── tnn_types.h
│   │   ├── tnn_utils.cpp
│   │   └── tnn_utils.h
│   └── cv    #  各个模型的具体实现，会引用core中实现的父类和功能函数
│       ├── tnn_age_googlenet.cpp
│       ├── tnn_age_googlenet.h
│       ├── tnn_cava_combined_face.cpp
│       ├── tnn_cava_combined_face.h
│       ├── tnn_cava_ghost_arcface.cpp
│       ├── tnn_cava_ghost_arcface.h
│       ├── tnn_center_loss_face.cpp

```
  
## 添加模型的步骤
以下以添加YOLOX的ONNXRuntime C++版本为例，讲解如何添加一个新模型。
* 第一步: 在 lite/ort/core/ort_core.h 中添加YoloX函数签名，如果是其他推理引擎则还需要加具体的推理引擎作为前缀，如MNNYoloX.
```C++
// lite/ort/core/ort_core.h 中
namespace ortcv
{
  // ... 
  class LITE_EXPORTS YoloX;                      // [56] * reference: https://github.com/Megvii-BaseDetection/YOLOX
}

// lite/mnn/core/mnn_core.h 中
namespace mnncv
{
// ... 
class LITE_EXPORTS MNNYoloX;                      // [3] * reference: https://github.com/Megvii-BaseDetection/YOLOX
}
```

* 第二步: 在 lite/ort/cv 中新建 yolox.h 和 yolox.cpp，注意最好文件名和你在xxx_core.h中的保持一致，方便管理，对于非ONNXRuntime的版本，还应该加上推理引擎作为前缀，如 mnn_yolox.h 和 mnn_yolox.cpp. 
```text
├── ort
│   ├── core     #  ONNXRuntime基础父类和特定功能的实现，必须要阅读
│   │   ├── ort_config.h
│...
│   │   └── ort_utils.h
│   └── cv      #  各个模型的具体实现，会引用core中实现的父类和功能函数
│       ├── yolox.cpp
│       ├── yolox.h
```
* 第三步: 编写 YoloX 类的头文件，由于是静态维度推理，并且是单输入多(单)输出模型，所以可以继承 BasicOrtHandler(请自行阅读lite/ort/core/ort_handler.cpp具体实现)，注意BasicOrtHandler有个transform的虚函数是要重写的。另外对于最终public的接口，也请保持detect的命名规范以确保API语义的统一性。请务必保持detect、detect_video等的命名规范，detect是图片级别的检测接口，detect_video是视频级别的检测接口。对于types命名空间，实际上对是全局types的引用，因此请在 lite/types.h中查看是否有合适的类型，如果没有则需要在 lite::types 命名空间中添加，然后再在 yolox.h 中使用，请尽可能保持自定义类型的简洁性。
```c++
#ifndef LITE_AI_ORT_CV_YOLOX_H
#define LITE_AI_ORT_CV_YOLOX_H

#include "lite/ort/core/ort_core.h"

namespace ortcv
{
  class LITE_EXPORTS YoloX : public BasicOrtHandler
  {
  public:
    explicit YoloX(const std::string &_onnx_path, unsigned int _num_threads = 1) :
        BasicOrtHandler(_onnx_path, _num_threads)
    {};

    ~YoloX() override = default;

  private:
    // nested classes
    typedef struct GridAndStride
    {
      int grid0;
      int grid1;
      int stride;
    } YoloXAnchor;

    typedef struct
    {
      float r;
      int dw;
      int dh;
      int new_unpad_w;
      int new_unpad_h;
      bool flag;
    } YoloXScaleParams;

  private:
    const float mean_vals[3] = {255.f * 0.485f, 255.f * 0.456, 255.f * 0.406f};
    const float scale_vals[3] = {1 / (255.f * 0.229f), 1 / (255.f * 0.224f), 1 / (255.f * 0.225f)};

    const char *class_names[80] = {
        "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat", "traffic light",
        "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow",
        "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
        "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard",
        "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
        "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch",
        "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote", "keyboard",
        "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase",
        "scissors", "teddy bear", "hair drier", "toothbrush"
    };
    enum NMS
    {
      HARD = 0, BLEND = 1, OFFSET = 2
    };
    static constexpr const unsigned int max_nms = 30000;

  private:
    // 需要被重写的方法
    Ort::Value transform(const cv::Mat &mat_rs) override; // without resize

    void resize_unscale(const cv::Mat &mat,
                        cv::Mat &mat_rs,
                        int target_height,
                        int target_width,
                        YoloXScaleParams &scale_params);

    void generate_anchors(const int target_height,
                          const int target_width,
                          std::vector<int> &strides,
                          std::vector<YoloXAnchor> &anchors);

    void generate_bboxes(const YoloXScaleParams &scale_params,
                         std::vector<types::Boxf> &bbox_collection,
                         std::vector<Ort::Value> &output_tensors,
                         float score_threshold, int img_height,
                         int img_width); // rescale & exclude

    void nms(std::vector<types::Boxf> &input, std::vector<types::Boxf> &output,
             float iou_threshold, unsigned int topk, unsigned int nms_type);

  public:
    // 请保持detect、detect_video等的命名规范，detect是图片级别的检测接口，detect_video是视频级别的检测接口
    void detect(const cv::Mat &mat, std::vector<types::Boxf> &detected_boxes,
                float score_threshold = 0.25f, float iou_threshold = 0.45f,
                unsigned int topk = 100, unsigned int nms_type = NMS::OFFSET);

  };
}

#endif //LITE_AI_ORT_CV_YOLOX_H
```
* 第四步: 在 yolox.cpp 中实现 YoloX 类的所有方法，并可能会引用全局的 lite/utils.h 进行复用，这基本是唯一的全局依赖了。
```c++
#include "yolox.h"
#include "lite/ort/core/ort_utils.h" // 引入onnxruntime特定的自定义功能函数，依赖于推理引擎
#include "lite/utils.h" // 引入全局定义的功能函数，不依赖推理引擎，如NMS

using ortcv::YoloX;

Ort::Value YoloX::transform(const cv::Mat &mat_rs)
{
  cv::Mat canvas;
  cv::cvtColor(mat_rs, canvas, cv::COLOR_BGR2RGB);
  // resize without padding, (Done): add padding as the official Python implementation.
  // cv::resize(canva, canva, cv::Size(input_node_dims.at(3),
  //                                  input_node_dims.at(2)));
  // (1,3,640,640) 1xCXHXW
  ortcv::utils::transform::normalize_inplace(canvas, mean_vals, scale_vals); // float32
  // Note !!!: Comment out this line if you use the newest YOLOX model.
  // There is no normalization for the newest official C++ implementation
  // using ncnn. Reference:
  // [1] https://github.com/Megvii-BaseDetection/YOLOX/blob/main/demo/ncnn/cpp/yolox.cpp
  // ortcv::utils::transform::normalize_inplace(canva, mean_vals, scale_vals); // float32
  return ortcv::utils::transform::create_tensor(
      canvas, input_node_dims, memory_info_handler,
      input_values_handler, ortcv::utils::transform::CHW);
}

void YoloX::resize_unscale(const cv::Mat &mat, cv::Mat &mat_rs,
                           int target_height, int target_width,
                           YoloXScaleParams &scale_params)
{
  if (mat.empty()) return;
  int img_height = static_cast<int>(mat.rows);
  int img_width = static_cast<int>(mat.cols);

  mat_rs = cv::Mat(target_height, target_width, CV_8UC3,
                   cv::Scalar(114, 114, 114));
  // scale ratio (new / old) new_shape(h,w)
  float w_r = (float) target_width / (float) img_width;
  float h_r = (float) target_height / (float) img_height;
  float r = std::min(w_r, h_r);
  // compute padding
  int new_unpad_w = static_cast<int>((float) img_width * r); // floor
  int new_unpad_h = static_cast<int>((float) img_height * r); // floor
  int pad_w = target_width - new_unpad_w; // >=0
  int pad_h = target_height - new_unpad_h; // >=0

  int dw = pad_w / 2;
  int dh = pad_h / 2;

  // resize with unscaling
  cv::Mat new_unpad_mat = mat.clone();
  cv::resize(new_unpad_mat, new_unpad_mat, cv::Size(new_unpad_w, new_unpad_h));
  new_unpad_mat.copyTo(mat_rs(cv::Rect(dw, dh, new_unpad_w, new_unpad_h)));

  // record scale params.
  scale_params.r = r;
  scale_params.dw = dw;
  scale_params.dh = dh;
  scale_params.new_unpad_w = new_unpad_w;
  scale_params.new_unpad_h = new_unpad_h;
  scale_params.flag = true;
}

void YoloX::detect(const cv::Mat &mat, std::vector<types::Boxf> &detected_boxes,
                   float score_threshold, float iou_threshold,
                   unsigned int topk, unsigned int nms_type)
{
  if (mat.empty()) return;
  const int input_height = input_node_dims.at(2);
  const int input_width = input_node_dims.at(3);
  int img_height = static_cast<int>(mat.rows);
  int img_width = static_cast<int>(mat.cols);

  // resize & unscale
  cv::Mat mat_rs;
  YoloXScaleParams scale_params;
  this->resize_unscale(mat, mat_rs, input_height, input_width, scale_params);

  // 1. make input tensor
  Ort::Value input_tensor = this->transform(mat_rs);
  // 2. inference scores & boxes.
  auto output_tensors = ort_session->Run(
      Ort::RunOptions{nullptr}, input_node_names.data(),
      &input_tensor, 1, output_node_names.data(), num_outputs
  );
  // 3. rescale & exclude.
  std::vector<types::Boxf> bbox_collection;
  this->generate_bboxes(scale_params, bbox_collection, output_tensors, score_threshold, img_height, img_width);
  // 4. hard|blend|offset nms with topk.
  this->nms(bbox_collection, detected_boxes, iou_threshold, topk, nms_type);
}

void YoloX::generate_anchors(const int target_height,
                             const int target_width,
                             std::vector<int> &strides,
                             std::vector<YoloXAnchor> &anchors)
{
  for (auto stride : strides)
  {
    int num_grid_w = target_width / stride;
    int num_grid_h = target_height / stride;
    for (int g1 = 0; g1 < num_grid_h; ++g1)
    {
      for (int g0 = 0; g0 < num_grid_w; ++g0)
      {
#ifdef LITE_WIN32
        YoloXAnchor anchor;
        anchor.grid0 = g0;
        anchor.grid1 = g1;
        anchor.stride = stride;
        anchors.push_back(anchor);
#else
        anchors.push_back((YoloXAnchor) {g0, g1, stride});
#endif
      }
    }
  }
}


void YoloX::generate_bboxes(const YoloXScaleParams &scale_params,
                            std::vector<types::Boxf> &bbox_collection,
                            std::vector<Ort::Value> &output_tensors,
                            float score_threshold, int img_height,
                            int img_width)
{
  Ort::Value &pred = output_tensors.at(0); // (1,n,85=5+80=cxcy+cwch+obj_conf+cls_conf)
  auto pred_dims = output_node_dims.at(0); // (1,n,85)
  const unsigned int num_anchors = pred_dims.at(1); // n = ?
  const unsigned int num_classes = pred_dims.at(2) - 5;
  const float input_height = static_cast<float>(input_node_dims.at(2)); // e.g 640
  const float input_width = static_cast<float>(input_node_dims.at(3)); // e.g 640

  std::vector<YoloXAnchor> anchors;
  std::vector<int> strides = {8, 16, 32}; // might have stride=64
  this->generate_anchors(input_height, input_width, strides, anchors);

  float r_ = scale_params.r;
  int dw_ = scale_params.dw;
  int dh_ = scale_params.dh;

  bbox_collection.clear();
  unsigned int count = 0;
  for (unsigned int i = 0; i < num_anchors; ++i)
  {
    float obj_conf = pred.At<float>({0, i, 4});
    if (obj_conf < score_threshold) continue; // filter first.

    float cls_conf = pred.At<float>({0, i, 5});
    unsigned int label = 0;
    for (unsigned int j = 0; j < num_classes; ++j)
    {
      float tmp_conf = pred.At<float>({0, i, j + 5});
      if (tmp_conf > cls_conf)
      {
        cls_conf = tmp_conf;
        label = j;
      }
    } // argmax
    float conf = obj_conf * cls_conf; // cls_conf (0.,1.)
    if (conf < score_threshold) continue; // filter

    const int grid0 = anchors.at(i).grid0;
    const int grid1 = anchors.at(i).grid1;
    const int stride = anchors.at(i).stride;

    float dx = pred.At<float>({0, i, 0});
    float dy = pred.At<float>({0, i, 1});
    float dw = pred.At<float>({0, i, 2});
    float dh = pred.At<float>({0, i, 3});

    float cx = (dx + (float) grid0) * (float) stride;
    float cy = (dy + (float) grid1) * (float) stride;
    float w = std::exp(dw) * (float) stride;
    float h = std::exp(dh) * (float) stride;
    float x1 = ((cx - w / 2.f) - (float) dw_) / r_;
    float y1 = ((cy - h / 2.f) - (float) dh_) / r_;
    float x2 = ((cx + w / 2.f) - (float) dw_) / r_;
    float y2 = ((cy + h / 2.f) - (float) dh_) / r_;

    types::Boxf box;
    box.x1 = std::max(0.f, x1);
    box.y1 = std::max(0.f, y1);
    box.x2 = std::min(x2, (float) img_width);
    box.y2 = std::min(y2, (float) img_height);
    box.score = conf;
    box.label = label;
    box.label_text = class_names[label];
    box.flag = true;
    bbox_collection.push_back(box);

    count += 1; // limit boxes for nms.
    if (count > max_nms)
      break;
  }
#if LITEORT_DEBUG
  std::cout << "detected num_anchors: " << num_anchors << "\n";
  std::cout << "generate_bboxes num: " << bbox_collection.size() << "\n";
#endif
}


void YoloX::nms(std::vector<types::Boxf> &input, std::vector<types::Boxf> &output,
                float iou_threshold, unsigned int topk, unsigned int nms_type)
{
  if (nms_type == NMS::BLEND) lite::utils::blending_nms(input, output, iou_threshold, topk);
  else if (nms_type == NMS::OFFSET) lite::utils::offset_nms(input, output, iou_threshold, topk);
  else lite::utils::hard_nms(input, output, iou_threshold, topk);
}
```  
* 第五步: 在 lite/models.h 中添加类型别名，进行命名空间管理，其他的推理引擎版本的命名空间管理类似。
```c++
// ENABLE_ONNXRUNTIME
#ifdef ENABLE_ONNXRUNTIME
// ...
#include "lite/ort/cv/yolox.h"
#endif

// 默认版本
namespace lite 
{
  namespace cv 
  {
#ifdef BACKEND_ONNXRUNTIME
    typedef ortcv::YoloX _YoloX;
#endif
  }
    // 2. general object detection
  namespace detection
  {
#ifdef BACKEND_ONNXRUNTIME
    typedef _YoloX YoloX;
#endif
  }
}
// 还有个onnxruntime的命名空间也要添加
namespace lite
{
  namespace onnxruntime 
  {
    namespace cv
    {
      typedef ortcv::YoloX _ONNXYoloX;
    }
    // 2. general object detection
    namespace detection
    {
      typedef _ONNXYoloX YoloX;
    }
  }
}
```
* 第六步: 编写测试工程 examples/lite/cv/test_lite_yolox.cpp
```c++
#include "lite/lite.h"

static void test_default()
{
  std::string onnx_path = "../../../hub/onnx/cv/yolox_s.onnx";
  std::string test_img_path = "../../../examples/lite/resources/test_lite_yolox_1.jpg";
  std::string save_img_path = "../../../logs/test_lite_yolox_1.jpg";

  // 1. Test Default Engine ONNXRuntime
  lite::cv::detection::YoloX *yolox = new lite::cv::detection::YoloX(onnx_path); // default

  std::vector<lite::types::Boxf> detected_boxes;
  cv::Mat img_bgr = cv::imread(test_img_path);
  yolox->detect(img_bgr, detected_boxes);

  lite::utils::draw_boxes_inplace(img_bgr, detected_boxes);

  cv::imwrite(save_img_path, img_bgr);

  std::cout << "Default Version Detected Boxes Num: " << detected_boxes.size() << std::endl;

  delete yolox;

}

static void test_onnxruntime()
{
#ifdef ENABLE_ONNXRUNTIME
  std::string onnx_path = "../../../hub/onnx/cv/yolox_s.onnx";
  std::string test_img_path = "../../../examples/lite/resources/test_lite_yolox_2.jpg";
  std::string save_img_path = "../../../logs/test_lite_yolox_2.jpg";

  // 2. Test Specific Engine ONNXRuntime
  lite::onnxruntime::cv::detection::YoloX *yolox =
      new lite::onnxruntime::cv::detection::YoloX(onnx_path);

  std::vector<lite::types::Boxf> detected_boxes;
  cv::Mat img_bgr = cv::imread(test_img_path);
  yolox->detect(img_bgr, detected_boxes);

  lite::utils::draw_boxes_inplace(img_bgr, detected_boxes);

  cv::imwrite(save_img_path, img_bgr);

  std::cout << "ONNXRuntime Version Detected Boxes Num: " << detected_boxes.size() << std::endl;

  delete yolox;
#endif
}

static void test_mnn()
{
#ifdef ENABLE_MNN
  // ...
#endif
}

static void test_ncnn()
{
#ifdef ENABLE_NCNN
  // ...
#endif
}

static void test_tnn()
{
#ifdef ENABLE_TNN
  // ...
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
```

* 第六步: 在 examples/CMakeLists.txt中增加编译可执行文件的选项。
```cmake
# ...
add_lite_executable(lite_yolox cv)
```
* 第七步: 重新编译工程，并测试示例(Mac/Linux，Windows还需要手动拷贝编译好的lite.ai.toolkit.dll以及其他依赖库到 build/lite.ai.toolkit/bin)
```shell
sh ./build.sh && cd build/lite.ai.toolkit/bin && ./lite_yolox
```
* 提示：对于多输入多输出模型，不能继承BasicOrtHandler，需要单独的实现，请参考 rvm.h 和 rvm.cpp 的做法。

## 成为贡献者
如果上述步骤全部通过后，可以考虑往 dev 分支（注意是dev分支，不是main，提到main可能不会被merge哦）提交 PR，并且通过百度云盘或谷歌云盘共享你的模型文件。我会下载您的代码和模型，进行编译测试，通过后代码会合并到main分支。