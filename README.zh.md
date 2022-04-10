
<!---
## <p align="center"> 🍅🍅Lite.AI.ToolKit: 一个开箱即用的C++ AI模型工具箱</p>
--->

<div id="lite.ai.toolkit-Introduction"></div>  


![logo-v3](docs/resources/logo-v3.png)

<div align='center'>
  <img src=https://img.shields.io/badge/mac|linux|win-pass-brightgreen.svg >
  <img src=https://img.shields.io/badge/device-GPU/CPU-yellow.svg >
  <img src=https://img.shields.io/badge/license-GPLv3-blue.svg >
  <img src=https://img.shields.io/badge/onnxruntime-1.10.0-turquoise.svg >
  <img src=https://img.shields.io/badge/mnn-1.2.0-hotpink.svg >
  <img src=https://img.shields.io/badge/ncnn-1.0.21-orange.svg >
  <img src=https://img.shields.io/badge/tnn-0.3.0-blue.svg >
</div>

🛠**Lite.Ai.ToolKit**: 一个轻量级的`C++` AI模型工具箱，用户友好（还行吧），开箱即用。已经包括 **[80+](https://github.com/DefTruth/lite.ai.toolkit/tree/main/docs/hub/lite.ai.toolkit.hub.onnx.md)** 流行的开源模型。这是一个根据个人兴趣整理的C++工具箱，, 涵盖[目标检测](#lite.ai.toolkit-object-detection)、[人脸检测](#lite.ai.toolkit-face-detection)、[人脸识别](#lite.ai.toolkit-face-recognition)、[语义分割](#lite.ai.toolkit-segmentation)、[抠图](#lite.ai.toolkit-matting)等领域。详见 [Model Zoo](#lite.ai.toolkit-Model-Zoo) 和 [ONNX Hub](https://github.com/DefTruth/lite.ai.toolkit/tree/main/docs/hub/lite.ai.toolkit.hub.onnx.md) 、[MNN Hub](https://github.com/DefTruth/lite.ai.toolkit/tree/main/docs/hub/lite.ai.toolkit.hub.mnn.md) 、[TNN Hub](https://github.com/DefTruth/lite.ai.toolkit/tree/main/docs/hub/lite.ai.toolkit.hub.tnn.md) 、[NCNN Hub](https://github.com/DefTruth/lite.ai.toolkit/tree/main/docs/hub/lite.ai.toolkit.hub.ncnn.md). [若是有用，❤️不妨给个⭐️🌟支持一下吧，感谢支持~]

<div align='center'>
  <img src='logs/test_lite_yolov5_1.jpg' height="90px" width="90px">
  <img src='docs/resources/efficientdet_d0.jpg' height="90px" width="90px">
  <img src='docs/resources/street.jpg' height="90px" width="90px">
  <img src='logs/test_lite_ultraface.jpg' height="90px" width="90px">
  <img src='logs/test_lite_face_landmarks_1000.jpg' height="90px" width="90px">
  <img src='logs/test_lite_fsanet.jpg' height="90px" width="90px">
  <img src='logs/test_lite_deeplabv3_resnet101.jpg' height="90px" width="90px">
  <img src='logs/test_lite_fast_style_transfer_mosaic.jpg' height="90px" width="90px"> 
  <br>
  <img src='docs/resources/teslai.gif' height="90px" width="90px">
  <img src='docs/resources/tesla.gif' height="90px" width="90px">
  <img src='docs/resources/dance3i.gif' height="90px" width="90px">
  <img src='docs/resources/dance3.gif' height="90px" width="90px">  
  <img src='docs/resources/yolop1.png' height="90px" width="90px">
  <img src='docs/resources/yolop1.gif' height="90px" width="90px">
  <img src='docs/resources/yolop2.png' height="90px" width="90px">
  <img src='docs/resources/yolop2.gif' height="90px" width="90px">
</div>    

<p align="center"><a href="README.md">English</a> | 中文文档 | <a href=#lite.ai.toolkit-Build-MacOS>MacOS</a> | <a href=#lite.ai.toolkit-Build-Linux>Linux</a> | <a href=#lite.ai.toolkit-Build-Windows>Windows</a> </p>


## 核心特征👏👋
<div id="lite.ai.toolkit-Core-Features"></div>

* **用户友好，开箱即用。** 使用简单一致的调用语法，如**lite::cv::Type::Class**，详见[examples](#lite.ai.toolkit-Examples-for-Lite.AI.ToolKit).
* **少量依赖，构建容易。** 目前, 默认只依赖 **OpenCV** 和 **ONNXRuntime**，详见[build](#lite.ai.toolkit-Build-Lite.AI.ToolKit)。
* **众多的算法模块，且持续更新。** 目前，包括 10+ 算法模块、**[80+](https://github.com/DefTruth/lite.ai.toolkit/tree/main/docs/hub/lite.ai.toolkit.hub.onnx.md)** 流行的开源模型以及 **[500+](https://github.com/DefTruth/lite.ai.toolkit/tree/main/docs/hub/lite.ai.toolkit.hub.onnx.md)** 权重文件

## 引用 🎉🎉

如果您在自己的项目中使用了**Lite.Ai.ToolKit**，可考虑按以下方式进行引用。
```BibTeX
@misc{lite.ai.toolkit2021,
  title={lite.ai.toolkit: A lite C++ toolkit of awesome AI models.},
  url={https://github.com/DefTruth/lite.ai.toolkit},
  note={Open-source software available at https://github.com/DefTruth/lite.ai.toolkit},
  author={Yan Jun},
  year={2021}
}
```  

## 关于训练 🤓👀  
一个用于人脸关键点检测的训练和评估的工具箱已经开源，可通过pip一键安装，地址在[torchlm](https://github.com/DefTruth/torchlm).

## 预编译库 和 技术规划 ✅
![](docs/resources/lite.ai.toolkit-roadmap-v0.1.png)

目前，有一些预编译的MacOS(x64)和Linux(x64)下的lite.ai.toolkit动态库，可以直接从以下链接进行下载。Windows(x64)和Android下的预编译库，也会在最近发布出来。更多详情请参考[issues#48](https://github.com/DefTruth/lite.ai.toolkit/issues/48) . 更多可下载的的预编译库，请跳转到[releases](https://github.com/DefTruth/lite.ai.toolkit/releases) 查看。

* [x]  [lite0.1.1-osx10.15.x-ocv4.5.2-ffmpeg4.2.2-onnxruntime1.8.1.zip](https://github.com/DefTruth/lite.ai.toolkit/releases/download/v0.1.1/lite0.1.1-osx10.15.x-ocv4.5.2-ffmpeg4.2.2-onnxruntime1.8.1.zip)
* [x]  [lite0.1.1-osx10.15.x-ocv4.5.2-ffmpeg4.2.2-onnxruntime1.9.0.zip](https://github.com/DefTruth/lite.ai.toolkit/releases/download/v0.1.1/lite0.1.1-osx10.15.x-ocv4.5.2-ffmpeg4.2.2-onnxruntime1.9.0.zip)
* [x]  [lite0.1.1-osx10.15.x-ocv4.5.2-ffmpeg4.2.2-onnxruntime1.10.0.zip](https://github.com/DefTruth/lite.ai.toolkit/releases/download/v0.1.1/lite0.1.1-osx10.15.x-ocv4.5.2-ffmpeg4.2.2-onnxruntime1.10.0.zip)
* [x]  [lite0.1.1-ubuntu18.04-ocv4.5.2-ffmpeg4.2.2-onnxruntime1.8.1.zip](https://github.com/DefTruth/lite.ai.toolkit/releases/download/v0.1.1/lite0.1.1-ubuntu18.04-ocv4.5.2-ffmpeg4.2.2-onnxruntime1.8.1.zip)
* [x]  [lite0.1.1-ubuntu18.04-ocv4.5.2-ffmpeg4.2.2-onnxruntime1.9.0.zip](https://github.com/DefTruth/lite.ai.toolkit/releases/download/v0.1.1/lite0.1.1-ubuntu18.04-ocv4.5.2-ffmpeg4.2.2-onnxruntime1.9.0.zip)
* [x]  [lite0.1.1-ubuntu18.04-ocv4.5.2-ffmpeg4.2.2-onnxruntime1.10.0.zip](https://github.com/DefTruth/lite.ai.toolkit/releases/download/v0.1.1/lite0.1.1-ubuntu18.04-ocv4.5.2-ffmpeg4.2.2-onnxruntime1.10.0.zip)

在Linux下，为了链接到预编译库，使用前，需要先将`lite.ai.toolkit/lib`的路径添加到LD_LIBRARY_PATH.
```shell
export LD_LIBRARY_PATH=YOUR-PATH-TO/lite.ai.toolkit/lib:$LD_LIBRARY_PATH
```
## 快速配置 👀

可以参考以下的CMakeLists.txt，快速配置lite.ai.toolkit。👇👀

```cmake
set(LITE_AI_DIR ${CMAKE_SOURCE_DIR}/lite.ai.toolkit)
include_directories(${LITE_AI_DIR}/include)
link_directories(${LITE_AI_DIR}/lib})
set(TOOLKIT_LIBS lite.ai.toolkit onnxruntime)
set(OpenCV_LIBS opencv_core opencv_imgcodecs opencv_imgproc opencv_video opencv_videoio)

add_executable(lite_yolov5 examples/test_lite_yolov5.cpp)
target_link_libraries(lite_yolov5 ${TOOLKIT_LIBS} ${OpenCV_LIBS})
```

## 目录 📖💡
* [核心特征](#lite.ai.toolkit-Core-Features)
* [快速开始](#lite.ai.toolkit-Quick-Start)
* [技术规划](#lite.ai.toolkit-RoadMap)
* [重要更新](#lite.ai.toolkit-Important-Updates)
* [模型支持矩阵](#lite.ai.toolkit-Supported-Models-Matrix)
* [编译文档](#lite.ai.toolkit-Build-Lite.AI.ToolKit)
* [模型下载](#lite.ai.toolkit-Model-Zoo)
* [应用案例](#lite.ai.toolkit-Examples-for-Lite.AI.ToolKit)
* [开源协议](#lite.ai.toolkit-License)
* [引用参考](#lite.ai.toolkit-References)
* [如何添加您的模型](#lite.ai.toolkit-Contribute)

## 1. 快速开始 🌟🌟
<div id="lite.ai.toolkit-Quick-Start"></div>

#### 案例0: 使用[YOLOv5](https://github.com/ultralytics/yolov5) 进行目标检测。请从Model-Zoo[<sup>2</sup>](#lite.ai.toolkit-2) 下载模型文件。
```c++
#include "lite/lite.h"

static void test_default()
{
  std::string onnx_path = "../../../hub/onnx/cv/yolov5s.onnx";
  std::string test_img_path = "../../../examples/lite/resources/test_lite_yolov5_1.jpg";
  std::string save_img_path = "../../../logs/test_lite_yolov5_1.jpg";

  auto *yolov5 = new lite::cv::detection::YoloV5(onnx_path); 
  std::vector<lite::types::Boxf> detected_boxes;
  cv::Mat img_bgr = cv::imread(test_img_path);
  yolov5->detect(img_bgr, detected_boxes);
  
  lite::utils::draw_boxes_inplace(img_bgr, detected_boxes);
  cv::imwrite(save_img_path, img_bgr);  
  
  delete yolov5;
}
```


## 2. 重要更新 🆕
<div id="lite.ai.toolkit-Important-Updates"></div>

<details> 
<summary> Click here to see details of Important Updates! </summary>

|     Date     |                                        Model                                         |                                                    C++                                                    |                         Paper                         |                                      Code                                       |                                        Awesome                                        |     Type     | 
|:------------:|:------------------------------------------------------------------------------------:|:---------------------------------------------------------------------------------------------------------:|:-----------------------------------------------------:|:-------------------------------------------------------------------------------:|:-------------------------------------------------------------------------------------:|:------------:|  
| 【2022/04/03】 |                      [MODNet](https://github.com/ZHKKKe/MODNet)                      |    [link](https://github.com/DefTruth/lite.ai.toolkit/blob/main/examples/lite/cv/test_lite_modnet.cpp)    |   [AAAI 2022](https://arxiv.org/pdf/2011.11961.pdf)   |                    [code](https://github.com/ZHKKKe/MODNet)                     |        ![](https://img.shields.io/github/stars/ZHKKKe/MODNet.svg?style=social)        |   matting    |
| 【2022/03/23】 |                   [PIPNtet](https://github.com/jhb86253817/PIPNet)                   |   [link](https://github.com/DefTruth/lite.ai.toolkit/blob/main/examples/lite/cv/test_lite_pipnet98.cpp)   |     [CVPR 2021](https://arxiv.org/abs/2003.03771)     |                  [code](https://github.com/jhb86253817/PIPNet)                  |     ![](https://img.shields.io/github/stars/jhb86253817/PIPNet.svg?style=social)      | face::align  |
| 【2022/01/19】 |                [YOLO5Face](https://github.com/deepcam-cn/yolov5-face)                |  [link](https://github.com/DefTruth/lite.ai.toolkit/blob/main/examples/lite/cv/test_lite_yolo5face.cpp)   |    [arXiv 2021](https://arxiv.org/abs/2105.12931)     |                [code](https://github.com/deepcam-cn/yolov5-face)                |   ![](https://img.shields.io/github/stars/deepcam-cn/yolov5-face.svg?style=social)    | face::detect |
| 【2022/01/07】 |   [SCRFD](https://github.com/deepinsight/insightface/blob/master/detection/scrfd/)   |    [link](https://github.com/DefTruth/lite.ai.toolkit/blob/main/examples/lite/cv/test_lite_scrfd.cpp)     |     [CVPR 2021](https://arxiv.org/abs/2105.04714)     | [code](https://github.com/deepinsight/insightface/blob/master/detection/scrfd/) |   ![](https://img.shields.io/github/stars/deepinsight/insightface.svg?style=social)   | face::detect |
| 【2021/12/27】 |                  [NanoDetPlus](https://github.com/RangiLyu/nanodet)                  | [link](https://github.com/DefTruth/lite.ai.toolkit/blob/main/examples/lite/cv/test_lite_nanodet_plus.cpp) |    [blog](https://zhuanlan.zhihu.com/p/449912627)     |                   [code](https://github.com/RangiLyu/nanodet)                   |      ![](https://img.shields.io/github/stars/RangiLyu/nanodet.svg?style=social)       |  detection   |
| 【2021/12/08】 |                 [MGMatting](https://github.com/yucornetto/MGMatting)                 |  [link](https://github.com/DefTruth/lite.ai.toolkit/blob/main/examples/lite/cv/test_lite_mg_matting.cpp)  |     [CVPR 2021](https://arxiv.org/abs/2012.06722)     |                 [code](https://github.com/yucornetto/MGMatting)                 |    ![](https://img.shields.io/github/stars/yucornetto/MGMatting.svg?style=social)     |   matting    |
| 【2021/11/11】 |       [YoloV5_V_6_0](https://github.com/ultralytics/yolov5/releases/tag/v6.0)        | [link](https://github.com/DefTruth/lite.ai.toolkit/blob/main/examples/lite/cv/test_lite_yolov5_v6.0.cpp)  | [doi](https://zenodo.org/record/5563715#.YbXffH1Bzfs) |         [code](https://github.com/ultralytics/yolov5/releases/tag/v6.0)         |     ![](https://img.shields.io/github/stars/ultralytics/yolov5.svg?style=social)      |  detection   |
| 【2021/10/26】 | [YoloX_V_0_1_1](https://github.com/Megvii-BaseDetection/YOLOX/releases/tag/0.1.1rc0) | [link](https://github.com/DefTruth/lite.ai.toolkit/blob/main/examples/lite/cv/test_lite_yolox_v0.1.1.cpp) |    [arXiv 2021](https://arxiv.org/abs/2107.08430)     |              [code](https://github.com/Megvii-BaseDetection/YOLOX)              | ![](https://img.shields.io/github/stars/Megvii-BaseDetection/YOLOX.svg?style=social)  |  detection   |
| 【2021/10/02】 |                    [NanoDet](https://github.com/RangiLyu/nanodet)                    |   [link](https://github.com/DefTruth/lite.ai.toolkit/blob/main/examples/lite/cv/test_lite_nanodet.cpp)    |    [blog](https://zhuanlan.zhihu.com/p/306530300)     |                   [code](https://github.com/RangiLyu/nanodet)                   |      ![](https://img.shields.io/github/stars/RangiLyu/nanodet.svg?style=social)       |  detection   |
| 【2021/09/20】 |         [RobustVideoMatting](https://github.com/PeterL1n/RobustVideoMatting)         |     [link](https://github.com/DefTruth/lite.ai.toolkit/blob/main/examples/lite/cv/test_lite_rvm.cpp)      |     [WACV 2022](https://arxiv.org/abs/2108.11515)     |             [code](https://github.com/PeterL1n/RobustVideoMatting)              | ![](https://img.shields.io/github/stars/PeterL1n/RobustVideoMatting.svg?style=social) |   matting    |
| 【2021/09/02】 |                       [YOLOP](https://github.com/hustvl/YOLOP)                       |    [link](https://github.com/DefTruth/lite.ai.toolkit/blob/main/examples/lite/cv/test_lite_yolop.cpp)     |    [arXiv 2021](https://arxiv.org/abs/2108.11250)     |                     [code](https://github.com/hustvl/YOLOP)                     |        ![](https://img.shields.io/github/stars/hustvl/YOLOP.svg?style=social)         |  detection   |

<!---
![](docs/resources/scrfd-mgmatting-nanodetplus.jpg)
--->

</details>

## 3. 模型支持矩阵
<div id="lite.ai.toolkit-Supported-Models-Matrix"></div>

* / = 暂不支持.
* ✅ = 可以运行，且官方支持.
* ✔️ = 可以运行，但非官方支持.
* ❔ = 计划中，但不会很快实现，也许几个月后.

|                                                       Class                                                       | Size  |       Type       |                                                          Demo                                                          | ONNXRuntime | MNN | NCNN | TNN | MacOS | Linux | Windows | Android |
|:-----------------------------------------------------------------------------------------------------------------:|:-----:|:----------------:|:----------------------------------------------------------------------------------------------------------------------:|:-----------:|:---:|:----:|:---:|:-----:|:-----:|:-------:|:-------:|
|                                  [YoloV5](https://github.com/ultralytics/yolov5)                                  |  28M  |   *detection*    |          [demo](https://github.com/DefTruth/lite.ai.toolkit/blob/main/examples/lite/cv/test_lite_yolov5.cpp)           |      ✅      |  ✅  |  ✅   |  ✅  |   ✅   |  ✔️   |   ✔️    |    ❔    |
|         [YoloV3](https://github.com/onnx/models/blob/master/vision/object_detection_segmentation/yolov3)          | 236M  |   *detection*    |          [demo](https://github.com/DefTruth/lite.ai.toolkit/blob/main/examples/lite/cv/test_lite_yolov3.cpp)           |      ✅      |  /  |  /   |  /  |   ✅   |  ✔️   |   ✔️    |    /    |
|     [TinyYoloV3](https://github.com/onnx/models/blob/master/vision/object_detection_segmentation/tiny-yolov3)     |  33M  |   *detection*    |        [demo](https://github.com/DefTruth/lite.ai.toolkit/blob/main/examples/lite/cv/test_lite_tiny_yolov3.cpp)        |      ✅      |  /  |  /   |  /  |   ✅   |  ✔️   |   ✔️    |    /    |
|                              [YoloV4](https://github.com/argusswift/YOLOv4-pytorch)                               | 176M  |   *detection*    |          [demo](https://github.com/DefTruth/lite.ai.toolkit/blob/main/examples/lite/cv/test_lite_yolov4.cpp)           |      ✅      |  /  |  /   |  /  |   ✅   |  ✔️   |   ✔️    |    /    |
|            [SSD](https://github.com/onnx/models/blob/master/vision/object_detection_segmentation/ssd)             |  76M  |   *detection*    |            [demo](https://github.com/DefTruth/lite.ai.toolkit/blob/main/examples/lite/cv/test_lite_ssd.cpp)            |      ✅      |  /  |  /   |  /  |   ✅   |  ✔️   |   ✔️    |    /    |
| [SSDMobileNetV1](https://github.com/onnx/models/blob/master/vision/object_detection_segmentation/ssd-mobilenetv1) |  27M  |   *detection*    |      [demo](https://github.com/DefTruth/lite.ai.toolkit/blob/main/examples/lite/cv/test_lite_ssd_mobilenetv1.cpp)      |      ✅      |  /  |  /   |  /  |   ✅   |  ✔️   |   ✔️    |    /    |
|                              [YoloX](https://github.com/Megvii-BaseDetection/YOLOX)                               | 3.5M  |   *detection*    |           [demo](https://github.com/DefTruth/lite.ai.toolkit/blob/main/examples/lite/cv/test_lite_yolox.cpp)           |      ✅      |  ✅  |  ✅   |  ✅  |   ✅   |  ✔️   |   ✔️    |    ❔    |
|                        [TinyYoloV4VOC](https://github.com/bubbliiiing/yolov4-tiny-pytorch)                        |  22M  |   *detection*    |      [demo](https://github.com/DefTruth/lite.ai.toolkit/blob/main/examples/lite/cv/test_lite_tiny_yolov4_voc.cpp)      |      ✅      |  /  |  /   |  /  |   ✅   |  ✔️   |   ✔️    |    /    |
|                       [TinyYoloV4COCO](https://github.com/bubbliiiing/yolov4-tiny-pytorch)                        |  22M  |   *detection*    |     [demo](https://github.com/DefTruth/lite.ai.toolkit/blob/main/examples/lite/cv/test_lite_tiny_yolov4_coco.cpp)      |      ✅      |  /  |  /   |  /  |   ✅   |  ✔️   |   ✔️    |    /    |
|                                   [YoloR](https://github.com/WongKinYiu/yolor)                                    |  39M  |   *detection*    |           [demo](https://github.com/DefTruth/lite.ai.toolkit/blob/main/examples/lite/cv/test_lite_yolor.cpp)           |      ✅      |  ✅  |  ✅   |  ✅  |   ✅   |  ✔️   |   ✔️    |    ❔    |
|                            [ScaledYoloV4](https://github.com/WongKinYiu/ScaledYOLOv4)                             | 270M  |   *detection*    |       [demo](https://github.com/DefTruth/lite.ai.toolkit/blob/main/examples/lite/cv/test_lite_scaled_yolov4.cpp)       |      ✅      |  /  |  /   |  /  |   ✅   |  ✔️   |   ✔️    |    /    |
|                    [EfficientDet](https://github.com/zylo117/Yet-Another-EfficientDet-Pytorch)                    |  15M  |   *detection*    |       [demo](https://github.com/DefTruth/lite.ai.toolkit/blob/main/examples/lite/cv/test_lite_efficientdet.cpp)        |      ✅      |  /  |  /   |  /  |   ✅   |  ✔️   |   ✔️    |    /    |
|                   [EfficientDetD7](https://github.com/zylo117/Yet-Another-EfficientDet-Pytorch)                   | 220M  |   *detection*    |      [demo](https://github.com/DefTruth/lite.ai.toolkit/blob/main/examples/lite/cv/test_lite_efficientdet_d7.cpp)      |      ✅      |  /  |  /   |  /  |   ✅   |  ✔️   |   ✔️    |    /    |
|                   [EfficientDetD8](https://github.com/zylo117/Yet-Another-EfficientDet-Pytorch)                   | 322M  |   *detection*    |      [demo](https://github.com/DefTruth/lite.ai.toolkit/blob/main/examples/lite/cv/test_lite_efficientdet_d8.cpp)      |      ✅      |  /  |  /   |  /  |   ✅   |  ✔️   |   ✔️    |    /    |
|                                     [YOLOP](https://github.com/hustvl/YOLOP)                                      |  30M  |   *detection*    |           [demo](https://github.com/DefTruth/lite.ai.toolkit/blob/main/examples/lite/cv/test_lite_yolop.cpp)           |      ✅      |  ✅  |  ✅   |  ✅  |   ✅   |  ✔️   |   ✔️    |    ❔    |
|                                  [NanoDet](https://github.com/RangiLyu/nanodet)                                   | 1.1M  |   *detection*    |          [demo](https://github.com/DefTruth/lite.ai.toolkit/blob/main/examples/lite/cv/test_lite_nanodet.cpp)          |      ✅      |  ✅  |  ✅   |  ✅  |   ✅   |  ✔️   |   ✔️    |    ❔    |
|                                [NanoDetPlus](https://github.com/RangiLyu/nanodet)                                 | 4.5M  |   *detection*    |       [demo](https://github.com/DefTruth/lite.ai.toolkit/blob/main/examples/lite/cv/test_lite_nanodet_plus.cpp)        |      ✅      |  ✅  |  ✅   |  ✅  |   ✅   |  ✔️   |   ✔️    |    ❔    |
|                               [NanoDetEffi...](https://github.com/RangiLyu/nanodet)                               |  12M  |   *detection*    | [demo](https://github.com/DefTruth/lite.ai.toolkit/blob/main/examples/lite/cv/test_lite_nanodet_efficientnet_lite.cpp) |      ✅      |  ✅  |  ✅   |  ✅  |   ✅   |  ✔️   |   ✔️    |    ❔    |
|                          [YoloX_V_0_1_1](https://github.com/Megvii-BaseDetection/YOLOX)                           | 3.5M  |   *detection*    |       [demo](https://github.com/DefTruth/lite.ai.toolkit/blob/main/examples/lite/cv/test_lite_yolox_v0.1.1.cpp)        |      ✅      |  ✅  |  ✅   |  ✅  |   ✅   |  ✔️   |   ✔️    |    ❔    |
|                               [YoloV5_V_6_0](https://github.com/ultralytics/yolov5)                               | 7.5M  |   *detection*    |        [demo](https://github.com/DefTruth/lite.ai.toolkit/blob/main/examples/lite/cv/test_lite_yolov5_v6.0.cpp)        |      ✅      |  ✅  |  ✅   |  ✅  |   ✅   |  ✔️   |   ✔️    |    ❔    |
|         [GlintArcFace](https://github.com/deepinsight/insightface/tree/master/recognition/arcface_torch)          |  92M  |     *faceid*     |       [demo](https://github.com/DefTruth/lite.ai.toolkit/blob/main/examples/lite/cv/test_lite_glint_arcface.cpp)       |      ✅      |  ✅  |  ✅   |  ✅  |   ✅   |  ✔️   |   ✔️    |    ❔    |
|         [GlintCosFace](https://github.com/deepinsight/insightface/tree/master/recognition/arcface_torch)          |  92M  |     *faceid*     |       [demo](https://github.com/DefTruth/lite.ai.toolkit/blob/main/examples/lite/cv/test_lite_glint_cosface.cpp)       |      ✅      |  ✅  |  ✅   |  ✅  |   ✅   |  ✔️   |   ✔️    |    /    |
|          [GlintPartialFC](https://github.com/deepinsight/insightface/tree/master/recognition/partial_fc)          | 170M  |     *faceid*     |     [demo](https://github.com/DefTruth/lite.ai.toolkit/blob/main/examples/lite/cv/test_lite_glint_partial_fc.cpp)      |      ✅      |  ✅  |  ✅   |  ✅  |   ✅   |  ✔️   |   ✔️    |    /    |
|                              [FaceNet](https://github.com/timesler/facenet-pytorch)                               |  89M  |     *faceid*     |          [demo](https://github.com/DefTruth/lite.ai.toolkit/blob/main/examples/lite/cv/test_lite_facenet.cpp)          |      ✅      |  ✅  |  ✅   |  ✅  |   ✅   |  ✔️   |   ✔️    |    /    |
|                         [FocalArcFace](https://github.com/ZhaoJ9014/face.evoLVe.PyTorch)                          | 166M  |     *faceid*     |       [demo](https://github.com/DefTruth/lite.ai.toolkit/blob/main/examples/lite/cv/test_lite_focal_arcface.cpp)       |      ✅      |  ✅  |  ✅   |  ✅  |   ✅   |  ✔️   |   ✔️    |    /    |
|                       [FocalAsiaArcFace](https://github.com/ZhaoJ9014/face.evoLVe.PyTorch)                        | 166M  |     *faceid*     |    [demo](https://github.com/DefTruth/lite.ai.toolkit/blob/main/examples/lite/cv/test_lite_focal_asia_arcface.cpp)     |      ✅      |  ✅  |  ✅   |  ✅  |   ✅   |  ✔️   |   ✔️    |    /    |
|                [TencentCurricularFace](https://github.com/Tencent/TFace/tree/master/tasks/distfc)                 | 249M  |     *faceid*     |  [demo](https://github.com/DefTruth/lite.ai.toolkit/blob/main/examples/lite/cv/test_lite_tencent_curricular_face.cpp)  |      ✅      |  ✅  |  ✅   |  ✅  |   ✅   |  ✔️   |   ✔️    |    /    |
|                    [TencentCifpFace](https://github.com/Tencent/TFace/tree/master/tasks/cifp)                     | 130M  |     *faceid*     |     [demo](https://github.com/DefTruth/lite.ai.toolkit/blob/main/examples/lite/cv/test_lite_tencent_cifp_face.cpp)     |      ✅      |  ✅  |  ✅   |  ✅  |   ✅   |  ✔️   |   ✔️    |    /    |
|                        [CenterLossFace](https://github.com/louis-she/center-loss.pytorch)                         | 280M  |     *faceid*     |     [demo](https://github.com/DefTruth/lite.ai.toolkit/blob/main/examples/lite/cv/test_lite_center_loss_face.cpp)      |      ✅      |  ✅  |  ✅   |  ✅  |   ✅   |  ✔️   |   ✔️    |    /    |
|                           [SphereFace](https://github.com/clcarwin/sphereface_pytorch)                            |  80M  |     *faceid*     |        [demo](https://github.com/DefTruth/lite.ai.toolkit/blob/main/examples/lite/cv/test_lite_sphere_face.cpp)        |      ✅      |  ✅  |  ✅   |  ✅  |   ✅   |  ✔️   |   ✔️    |    /    |
|                              [PoseRobustFace](https://github.com/penincillin/DREAM)                               |  92M  |     *faceid*     |     [demo](https://github.com/DefTruth/lite.ai.toolkit/blob/main/examples/lite/cv/test_lite_pose_robust_face.cpp)      |      ✅      |  /  |  /   |  /  |   ✅   |  ✔️   |   ✔️    |    /    |
|                            [NaivePoseRobustFace](https://github.com/penincillin/DREAM)                            |  43M  |     *faceid*     |  [demo](https://github.com/DefTruth/lite.ai.toolkit/blob/main/examples/lite/cv/test_lite_naive_pose_robust_face.cpp)   |      ✅      |  /  |  /   |  /  |   ✅   |  ✔️   |   ✔️    |    /    |
|                        [MobileFaceNet](https://github.com/Xiaoccer/MobileFaceNet_Pytorch)                         | 3.8M  |     *faceid*     |      [demo](https://github.com/DefTruth/lite.ai.toolkit/blob/main/examples/lite/cv/test_lite_mobile_facenet.cpp)       |      ✅      |  ✅  |  ✅   |  ✅  |   ✅   |  ✔️   |   ✔️    |    ❔    |
|                        [CavaGhostArcFace](https://github.com/cavalleria/cavaface.pytorch)                         |  15M  |     *faceid*     |    [demo](https://github.com/DefTruth/lite.ai.toolkit/blob/main/examples/lite/cv/test_lite_cava_ghost_arcface.cpp)     |      ✅      |  ✅  |  ✅   |  ✅  |   ✅   |  ✔️   |   ✔️    |    ❔    |
|                        [CavaCombinedFace](https://github.com/cavalleria/cavaface.pytorch)                         | 250M  |     *faceid*     |    [demo](https://github.com/DefTruth/lite.ai.toolkit/blob/main/examples/lite/cv/test_lite_cava_combined_face.cpp)     |      ✅      |  ✅  |  ✅   |  ✅  |   ✅   |  ✔️   |   ✔️    |    /    |
|                    [MobileSEFocalFace](https://github.com/grib0ed0v/face_recognition.pytorch)                     | 4.5M  |     *faceid*     |    [demo](https://github.com/DefTruth/lite.ai.toolkit/blob/main/examples/lite/cv/test_lite_mobilese_focal_face.cpp)    |      ✅      |  ✅  |  ✅   |  ✅  |   ✅   |  ✔️   |   ✔️    |    ❔    |
|                       [RobustVideoMatting](https://github.com/PeterL1n/RobustVideoMatting)                        |  14M  |    *matting*     |            [demo](https://github.com/DefTruth/lite.ai.toolkit/blob/main/examples/lite/cv/test_lite_rvm.cpp)            |      ✅      |  ✅  |  /   |  ✅  |   ✅   |  ✔️   |   ✔️    |    ❔    |
|                               [MGMatting](https://github.com/yucornetto/MGMatting)                                | 113M  |    *matting*     |        [demo](https://github.com/DefTruth/lite.ai.toolkit/blob/main/examples/lite/cv/test_lite_mg_matting.cpp)         |      ✅      |  ✅  |  /   |  ✅  |   ✅   |  ✔️   |   ✔️    |    /    |
|                                    [MODNet](https://github.com/ZHKKKe/MODNet)                                     |  24M  |    *matting*     |          [demo](https://github.com/DefTruth/lite.ai.toolkit/blob/main/examples/lite/cv/test_lite_modnet.cpp)           |      ✅      |  ✅  |  ✅   |  ✅  |   ✅   |  ✔️   |   ✔️    |    /    |
|                                   [MODNetDyn](https://github.com/ZHKKKe/MODNet)                                   |  24M  |    *matting*     |        [demo](https://github.com/DefTruth/lite.ai.toolkit/blob/main/examples/lite/cv/test_lite_modnet_dyn.cpp)         |      ✅      |  /  |  /   |  /  |   ✅   |  ✔️   |   ✔️    |    /    |
|                      [BackgroundMattingV2](https://github.com/PeterL1n/BackgroundMattingV2)                       |  20M  |    *matting*     |    [demo](https://github.com/DefTruth/lite.ai.toolkit/blob/main/examples/lite/cv/test_lite_backgroundmattingv2.cpp)    |      ✅      |  ✅  |  /   |  ✅  |   ✅   |  ✔️   |   ✔️    |    /    |
|                     [BackgroundMattingV2Dyn](https://github.com/PeterL1n/BackgroundMattingV2)                     |  20M  |    *matting*     |  [demo](https://github.com/DefTruth/lite.ai.toolkit/blob/main/examples/lite/cv/test_lite_backgroundmattingv2_dyn.cpp)  |      ✅      |  /  |  /   |  /  |   ✅   |  ✔️   |   ✔️    |    /    |
|                [UltraFace](https://github.com/Linzaer/Ultra-Light-Fast-Generic-Face-Detector-1MB)                 | 1.1M  |  *face::detect*  |         [demo](https://github.com/DefTruth/lite.ai.toolkit/blob/main/examples/lite/cv/test_lite_ultraface.cpp)         |      ✅      |  ✅  |  ✅   |  ✅  |   ✅   |  ✔️   |   ✔️    |    ❔    |
|                            [RetinaFace](https://github.com/biubug6/Pytorch_Retinaface)                            | 1.6M  |  *face::detect*  |        [demo](https://github.com/DefTruth/lite.ai.toolkit/blob/main/examples/lite/cv/test_lite_retinaface.cpp)         |      ✅      |  ✅  |  ✅   |  ✅  |   ✅   |  ✔️   |   ✔️    |    ❔    |
|                             [FaceBoxes](https://github.com/zisianw/FaceBoxes.PyTorch)                             | 3.8M  |  *face::detect*  |         [demo](https://github.com/DefTruth/lite.ai.toolkit/blob/main/examples/lite/cv/test_lite_faceboxes.cpp)         |      ✅      |  ✅  |  ✅   |  ✅  |   ✅   |  ✔️   |   ✔️    |    ❔    |
|                             [FaceBoxesV2](https://github.com/jhb86253817/FaceBoxesV2)                             | 3.8M  |  *face::detect*  |        [demo](https://github.com/DefTruth/lite.ai.toolkit/blob/main/examples/lite/cv/test_lite_faceboxesv2.cpp)        |      ✅      |  ✅  |  ✅   |  ✅  |   ✅   |  ✔️   |   ✔️    |    ❔    |
|                 [SCRFD](https://github.com/deepinsight/insightface/blob/master/detection/scrfd/)                  | 2.5M  |  *face::detect*  |           [demo](https://github.com/DefTruth/lite.ai.toolkit/blob/main/examples/lite/cv/test_lite_scrfd.cpp)           |      ✅      |  ✅  |  ✅   |  ✅  |   ✅   |  ✔️   |   ✔️    |    ❔    |  
|                              [YOLO5Face](https://github.com/deepcam-cn/yolov5-face)                               | 4.8M  |  *face::detect*  |         [demo](https://github.com/DefTruth/lite.ai.toolkit/blob/main/examples/lite/cv/test_lite_yolo5face.cpp)         |      ✅      |  ✅  |  ✅   |  ✅  |   ✅   |  ✔️   |   ✔️    |    ❔    |  
|                            [PFLD](https://github.com/Hsintao/pfld_106_face_landmarks)                             | 1.0M  |  *face::align*   |           [demo](https://github.com/DefTruth/lite.ai.toolkit/blob/main/examples/lite/cv/test_lite_pfld.cpp)            |      ✅      |  ✅  |  ✅   |  ✅  |   ✅   |  ✔️   |   ✔️    |    ❔    |
|                               [PFLD98](https://github.com/polarisZhao/PFLD-pytorch)                               | 4.8M  |  *face::align*   |          [demo](https://github.com/DefTruth/lite.ai.toolkit/blob/main/examples/lite/cv/test_lite_pfld98.cpp)           |      ✅      |  ✅  |  ✅   |  ✅  |   ✅   |  ✔️   |   ✔️    |    ❔    |
|                         [MobileNetV268](https://github.com/cunjian/pytorch_face_landmark)                         | 9.4M  |  *face::align*   |      [demo](https://github.com/DefTruth/lite.ai.toolkit/blob/main/examples/lite/cv/test_lite_mobilenetv2_68.cpp)       |      ✅      |  ✅  |  ✅   |  ✅  |   ✅   |  ✔️   |   ✔️    |    ❔    |
|                        [MobileNetV2SE68](https://github.com/cunjian/pytorch_face_landmark)                        |  11M  |  *face::align*   |     [demo](https://github.com/DefTruth/lite.ai.toolkit/blob/main/examples/lite/cv/test_lite_mobilenetv2_se_68.cpp)     |      ✅      |  ✅  |  ✅   |  ✅  |   ✅   |  ✔️   |   ✔️    |    ❔    |
|                            [PFLD68](https://github.com/cunjian/pytorch_face_landmark)                             | 2.8M  |  *face::align*   |          [demo](https://github.com/DefTruth/lite.ai.toolkit/blob/main/examples/lite/cv/test_lite_pfld68.cpp)           |      ✅      |  ✅  |  ✅   |  ✅  |   ✅   |  ✔️   |   ✔️    |    ❔    |
|                         [FaceLandmark1000](https://github.com/Single430/FaceLandmark1000)                         | 2.0M  |  *face::align*   |    [demo](https://github.com/DefTruth/lite.ai.toolkit/blob/main/examples/lite/cv/test_lite_face_landmarks_1000.cpp)    |      ✅      |  ✅  |  ✅   |  ✅  |   ✅   |  ✔️   |   ✔️    |    ❔    |
|                                 [PIPNet98](https://github.com/jhb86253817/PIPNet)                                 | 44.0M |  *face::align*   |         [demo](https://github.com/DefTruth/lite.ai.toolkit/blob/main/examples/lite/cv/test_lite_pipnet98.cpp)          |      ✅      |  ✅  |  ✅   |  ✅  |   ✅   |  ✔️   |   ✔️    |    ❔    |
|                                 [PIPNet68](https://github.com/jhb86253817/PIPNet)                                 | 44.0M |  *face::align*   |         [demo](https://github.com/DefTruth/lite.ai.toolkit/blob/main/examples/lite/cv/test_lite_pipnet68.cpp)          |      ✅      |  ✅  |  ✅   |  ✅  |   ✅   |  ✔️   |   ✔️    |    ❔    |
|                                 [PIPNet29](https://github.com/jhb86253817/PIPNet)                                 | 44.0M |  *face::align*   |         [demo](https://github.com/DefTruth/lite.ai.toolkit/blob/main/examples/lite/cv/test_lite_pipnet29.cpp)          |      ✅      |  ✅  |  ✅   |  ✅  |   ✅   |  ✔️   |   ✔️    |    ❔    |
|                                 [PIPNet19](https://github.com/jhb86253817/PIPNet)                                 | 44.0M |  *face::align*   |         [demo](https://github.com/DefTruth/lite.ai.toolkit/blob/main/examples/lite/cv/test_lite_pipnet19.cpp)          |      ✅      |  ✅  |  ✅   |  ✅  |   ✅   |  ✔️   |   ✔️    |    ❔    |
|                           [FSANet](https://github.com/omasaht/headpose-fsanet-pytorch)                            | 1.2M  |   *face::pose*   |          [demo](https://github.com/DefTruth/lite.ai.toolkit/blob/main/examples/lite/cv/test_lite_fsanet.cpp)           |      ✅      |  ✅  |  /   |  ✅  |   ✅   |  ✔️   |   ✔️    |    ❔    |
|            [AgeGoogleNet](https://github.com/onnx/models/tree/master/vision/body_analysis/age_gender)             |  23M  |   *face::attr*   |       [demo](https://github.com/DefTruth/lite.ai.toolkit/blob/main/examples/lite/cv/test_lite_age_googlenet.cpp)       |      ✅      |  ✅  |  ✅   |  ✅  |   ✅   |  ✔️   |   ✔️    |    ❔    |
|           [GenderGoogleNet](https://github.com/onnx/models/tree/master/vision/body_analysis/age_gender)           |  23M  |   *face::attr*   |     [demo](https://github.com/DefTruth/lite.ai.toolkit/blob/main/examples/lite/cv/test_lite_gender_googlenet.cpp)      |      ✅      |  ✅  |  ✅   |  ✅  |   ✅   |  ✔️   |   ✔️    |    ❔    |
|         [EmotionFerPlus](https://github.com/onnx/models/blob/master/vision/body_analysis/emotion_ferplus)         |  33M  |   *face::attr*   |      [demo](https://github.com/DefTruth/lite.ai.toolkit/blob/main/examples/lite/cv/test_lite_emotion_ferplus.cpp)      |      ✅      |  ✅  |  ✅   |  ✅  |   ✅   |  ✔️   |   ✔️    |    ❔    |
|              [VGG16Age](https://github.com/onnx/models/tree/master/vision/body_analysis/age_gender)               | 514M  |   *face::attr*   |         [demo](https://github.com/DefTruth/lite.ai.toolkit/blob/main/examples/lite/cv/test_lite_vgg16_age.cpp)         |      ✅      |  ✅  |  ✅   |  ✅  |   ✅   |  ✔️   |   ✔️    |    /    |
|             [VGG16Gender](https://github.com/onnx/models/tree/master/vision/body_analysis/age_gender)             | 512M  |   *face::attr*   |       [demo](https://github.com/DefTruth/lite.ai.toolkit/blob/main/examples/lite/cv/test_lite_vgg16_gender.cpp)        |      ✅      |  ✅  |  ✅   |  ✅  |   ✅   |  ✔️   |   ✔️    |    /    |
|                               [SSRNet](https://github.com/oukohou/SSR_Net_Pytorch)                                | 190K  |   *face::attr*   |          [demo](https://github.com/DefTruth/lite.ai.toolkit/blob/main/examples/lite/cv/test_lite_ssrnet.cpp)           |      ✅      |  ✅  |  /   |  ✅  |   ✅   |  ✔️   |   ✔️    |    ❔    |
|                  [EfficientEmotion7](https://github.com/HSE-asavchenko/face-emotion-recognition)                  |  15M  |   *face::attr*   |    [demo](https://github.com/DefTruth/lite.ai.toolkit/blob/main/examples/lite/cv/test_lite_efficient_emotion7.cpp)     |      ✅      |  ✅  |  ✅   |  ✅  |   ✅   |  ✔️   |   ✔️    |    ❔    |
|                  [EfficientEmotion8](https://github.com/HSE-asavchenko/face-emotion-recognition)                  |  15M  |   *face::attr*   |    [demo](https://github.com/DefTruth/lite.ai.toolkit/blob/main/examples/lite/cv/test_lite_efficient_emotion8.cpp)     |      ✅      |  ✅  |  ✅   |  ✅  |   ✅   |  ✔️   |   ✔️    |    ❔    |
|                   [MobileEmotion7](https://github.com/HSE-asavchenko/face-emotion-recognition)                    |  13M  |   *face::attr*   |      [demo](https://github.com/DefTruth/lite.ai.toolkit/blob/main/examples/lite/cv/test_lite_mobile_emotion7.cpp)      |      ✅      |  ✅  |  ✅   |  ✅  |   ✅   |  ✔️   |   ✔️    |    ❔    |
|                   [ReXNetEmotion7](https://github.com/HSE-asavchenko/face-emotion-recognition)                    |  30M  |   *face::attr*   |      [demo](https://github.com/DefTruth/lite.ai.toolkit/blob/main/examples/lite/cv/test_lite_rexnet_emotion7.cpp)      |      ✅      |  ✅  |  /   |  ✅  |   ✅   |  ✔️   |   ✔️    |    /    |
|     [EfficientNetLite4](https://github.com/onnx/models/blob/master/vision/classification/efficientnet-lite4)      |  49M  | *classification* |    [demo](https://github.com/DefTruth/lite.ai.toolkit/blob/main/examples/lite/cv/test_lite_efficientnet_lite4.cpp)     |      ✅      |  ✅  |  /   |  ✅  |   ✅   |  ✔️   |   ✔️    |    /    |
|            [ShuffleNetV2](https://github.com/onnx/models/blob/master/vision/classification/shufflenet)            | 8.7M  | *classification* |       [demo](https://github.com/DefTruth/lite.ai.toolkit/blob/main/examples/lite/cv/test_lite_shufflenetv2.cpp)        |      ✅      |  ✅  |  ✅   |  ✅  |   ✅   |  ✔️   |   ✔️    |    ❔    |
|                          [DenseNet121](https://pytorch.org/hub/pytorch_vision_densenet/)                          | 30.7M | *classification* |         [demo](https://github.com/DefTruth/lite.ai.toolkit/blob/main/examples/lite/cv/test_lite_densenet.cpp)          |      ✅      |  ✅  |  ✅   |  ✅  |   ✅   |  ✔️   |   ✔️    |    /    |
|                           [GhostNet](https://pytorch.org/hub/pytorch_vision_ghostnet/)                            |  20M  | *classification* |         [demo](https://github.com/DefTruth/lite.ai.toolkit/blob/main/examples/lite/cv/test_lite_ghostnet.cpp)          |      ✅      |  ✅  |  ✅   |  ✅  |   ✅   |  ✔️   |   ✔️    |    ❔    |
|                            [HdrDNet](https://pytorch.org/hub/pytorch_vision_hardnet//)                            |  13M  | *classification* |          [demo](https://github.com/DefTruth/lite.ai.toolkit/blob/main/examples/lite/cv/test_lite_hardnet.cpp)          |      ✅      |  ✅  |  ✅   |  ✅  |   ✅   |  ✔️   |   ✔️    |    ❔    |
|                             [IBNNet](https://pytorch.org/hub/pytorch_vision_ibnnet/)                              |  97M  | *classification* |          [demo](https://github.com/DefTruth/lite.ai.toolkit/blob/main/examples/lite/cv/test_lite_ibnnet.cpp)           |      ✅      |  ✅  |  ✅   |  ✅  |   ✅   |  ✔️   |   ✔️    |    /    |
|                        [MobileNetV2](https://pytorch.org/hub/pytorch_vision_mobilenet_v2/)                        |  13M  | *classification* |        [demo](https://github.com/DefTruth/lite.ai.toolkit/blob/main/examples/lite/cv/test_lite_mobilenetv2.cpp)        |      ✅      |  ✅  |  ✅   |  ✅  |   ✅   |  ✔️   |   ✔️    |    ❔    |
|                             [ResNet](https://pytorch.org/hub/pytorch_vision_resnet/)                              |  44M  | *classification* |          [demo](https://github.com/DefTruth/lite.ai.toolkit/blob/main/examples/lite/cv/test_lite_resnet.cpp)           |      ✅      |  ✅  |  ✅   |  ✅  |   ✅   |  ✔️   |   ✔️    |    /    |
|                            [ResNeXt](https://pytorch.org/hub/pytorch_vision_resnext/)                             |  95M  | *classification* |          [demo](https://github.com/DefTruth/lite.ai.toolkit/blob/main/examples/lite/cv/test_lite_resnext.cpp)          |      ✅      |  ✅  |  ✅   |  ✅  |   ✅   |  ✔️   |   ✔️    |    /    |
|                 [DeepLabV3ResNet101](https://pytorch.org/hub/pytorch_vision_deeplabv3_resnet101/)                 | 232M  |  *segmentation*  |    [demo](https://github.com/DefTruth/lite.ai.toolkit/blob/main/examples/lite/cv/test_lite_deeplabv3_resnet101.cpp)    |      ✅      |  ✅  |  ✅   |  ✅  |   ✅   |  ✔️   |   ✔️    |    /    |
|                       [FCNResNet101](https://pytorch.org/hub/pytorch_vision_fcn_resnet101/)                       | 207M  |  *segmentation*  |       [demo](https://github.com/DefTruth/lite.ai.toolkit/blob/main/examples/lite/cv/test_lite_fcn_resnet101.cpp)       |      ✅      |  ✅  |  ✅   |  ✅  |   ✅   |  ✔️   |   ✔️    |    /    |
|      [FastStyleTransfer](https://github.com/onnx/models/blob/master/vision/style_transfer/fast_neural_style)      | 6.4M  |     *style*      |    [demo](https://github.com/DefTruth/lite.ai.toolkit/blob/main/examples/lite/cv/test_lite_fast_style_transfer.cpp)    |      ✅      |  ✅  |  ✅   |  ✅  |   ✅   |  ✔️   |   ✔️    |    ❔    |
|                              [Colorizer](https://github.com/richzhang/colorization)                               | 123M  |  *colorization*  |         [demo](https://github.com/DefTruth/lite.ai.toolkit/blob/main/examples/lite/cv/test_lite_colorizer.cpp)         |      ✅      |  ✅  |  /   |  ✅  |   ✅   |  ✔️   |   ✔️    |    /    |
|                             [SubPixelCNN](https://github.com/niazwazir/SUB_PIXEL_CNN)                             | 234K  |   *resolution*   |       [demo](https://github.com/DefTruth/lite.ai.toolkit/blob/main/examples/lite/cv/test_lite_subpixel_cnn.cpp)        |      ✅      |  ✅  |  /   |  ✅  |   ✅   |  ✔️   |   ✔️    |    ❔    |
|                             [SubPixelCNN](https://github.com/niazwazir/SUB_PIXEL_CNN)                             | 234K  |   *resolution*   |       [demo](https://github.com/DefTruth/lite.ai.toolkit/blob/main/examples/lite/cv/test_lite_subpixel_cnn.cpp)        |      ✅      |  ✅  |  /   |  ✅  |   ✅   |  ✔️   |   ✔️    |    ❔    |
|                           [InsectDet](https://github.com/quarrying/quarrying-insect-id)                           |  27M  |   *detection*    |         [demo](https://github.com/DefTruth/lite.ai.toolkit/blob/main/examples/lite/cv/test_lite_insectdet.cpp)         |      ✅      |  ✅  |  /   |  ✅  |   ✅   |  ✔️   |   ✔️    |    ❔    |
|                           [InsectID](https://github.com/quarrying/quarrying-insect-id)                            |  22M  | *classification* |         [demo](https://github.com/DefTruth/lite.ai.toolkit/blob/main/examples/lite/cv/test_lite_insectid.cpp)          |      ✅      |  ✅  |  ✅   |  ✅  |   ✅   |   ✅   |   ✔️    |   ✔️    |    ❔    |
|                            [PlantID](https://github.com/quarrying/quarrying-plant-id)                             |  30M  | *classification* |          [demo](https://github.com/DefTruth/lite.ai.toolkit/blob/main/examples/lite/cv/test_lite_plantid.cpp)          |      ✅      |  ✅  |  ✅   |  ✅  |   ✅   |   ✅   |   ✔️    |   ✔️    |    ❔    |


## 4. 编译文档
<div id="lite.ai.toolkit-Build-MacOS"></div>
<div id="lite.ai.toolkit-Build-Lite.AI.ToolKit"></div>

* MacOS: 从**Lite.Ai.ToolKit** 源码编译**MacOS**下的动态库。需要注意的是**Lite.Ai.ToolKit** 使用**onnxruntime**作为默认的后端，因为onnxruntime支持大部分onnx的原生算子，具有更高的易用性。如何编译Linux和Windows版本？点击 ▶️ 查看。
```shell
    git clone --depth=1 https://github.com/DefTruth/lite.ai.toolkit.git  # 最新源码
    cd lite.ai.toolkit && sh ./build.sh  # 对于MacOS, 你可以直接利用本项目包含的OpenCV, ONNXRuntime, MNN, NCNN and TNN依赖库，无需重新编译
```

<div id="lite.ai.toolkit-Build-Linux"></div>
<div id="lite.ai.toolkit-Build-Windows"></div>

<details>
<summary>💡️ Linux 和 Windows </summary>  

### Linux 和 Windows

⚠️ **Lite.Ai.ToolKit** 的发行版本目前不直接支持Linux和Windows，你需要从下载**Lite.Ai.ToolKit**的源码进行构建。首先，你需要下载(如果有官方编译好的发行版本的话)或编译**OpenCV** 、**ONNXRuntime** 和其他你需要的推理引擎，如MNN、NCNN、TNN，然后把它们的头文件分别放入各自对应的文件夹，或者直接使用本项目提供的头文件。本项目的依赖库头文件是直接从相应的官方库拷贝而来的，但不同操作系统下的动态库需要重新编译或下载，MacOS用户可以直接使用本项目提供的各个依赖库的动态库。
* **lite.ai.toolkit/opencv2**
  ```shell
    cp -r you-path-to-downloaded-or-built-opencv/include/opencv4/opencv2 lite.ai.toolkit/opencv2
  ```
* **lite.ai.toolkit/onnxruntime**
  ```shell
    cp -r you-path-to-downloaded-or-built-onnxruntime/include/onnxruntime lite.ai.toolkit/onnxruntime
  ```
* **lite.ai.toolkit/MNN**
  ```shell
    cp -r you-path-to-downloaded-or-built-MNN/include/MNN lite.ai.toolkit/MNN
  ```
* **lite.ai.toolkit/ncnn**
  ```shell
    cp -r you-path-to-downloaded-or-built-ncnn/include/ncnn lite.ai.toolkit/ncnn
  ```
* **lite.ai.toolkit/tnn**
  ```shell
    cp -r you-path-to-downloaded-or-built-TNN/include/tnn lite.ai.toolkit/tnn
  ```

然后把各个依赖库拷贝到**lite.ai.toolkit/lib/(linux|windows)** 文件夹。 请参考依赖库的编译文档[<sup>1</sup>](#lite.ai.toolkit-1)。
* **lite.ai.toolkit/lib/(linux|windows)**
  ```shell
    cp you-path-to-downloaded-or-built-opencv/lib/*opencv* lite.ai.toolkit/lib/(linux|windows)/
    cp you-path-to-downloaded-or-built-onnxruntime/lib/*onnxruntime* lite.ai.toolkit/lib/(linux|windows)/
    cp you-path-to-downloaded-or-built-MNN/lib/*MNN* lite.ai.toolkit/lib/(linux|windows)/
    cp you-path-to-downloaded-or-built-ncnn/lib/*ncnn* lite.ai.toolkit/lib/(linux|windows)/
    cp you-path-to-downloaded-or-built-TNN/lib/*TNN* lite.ai.toolkit/lib/(linux|windows)/
  ```
注意，你还需要安装ffmpeg(<=4.2.2)，因为opencv的videoio模块依赖ffmpeg进行mp4的编解码。参考[issue#203](https://github.com/DefTruth/lite.ai.toolkit/issues/6) . 在MacOS下，ffmpeg4.2.2已经作为一个自定义依赖库被我打包进lite.ai.toolkit，不需要再从HomeBrew安装为系统库，因此lite.ai.toolkit是单体的，你可以把它移植到app里面，不用心另一台运行app的机器没有ffmpeg，MacOS版本的lite.ai.toolkit已经包含ffmpeg. 在Windows下，opencv官方团队已经提供了用于opencv的ffmpeg预编译库。在Linux下编译opencv时，请确保-DWITH_FFMPEG=ON，并检查是否链接到ffmpeg. 
* 先编译ffmpeg，注意必须是低版本的，高于4.4的，opencv会不兼容。
```shell
git clone --depth=1 https://git.ffmpeg.org/ffmpeg.git -b n4.2.2
cd ffmpeg
./configure --enable-shared --disable-x86asm --prefix=/usr/local/opt/ffmpeg --disable-static
make -j8
make install
```
* 然后，编译带ffmpeg支持的OpenCV，指定-DWITH_FFMPEG=ON
```shell
#!/bin/bash

mkdir build
cd build

cmake .. \
  -D CMAKE_BUILD_TYPE=Release \
  -D CMAKE_INSTALL_PREFIX=your-path-to-custom-dir \
  -D BUILD_TESTS=OFF \
  -D BUILD_PERF_TESTS=OFF \
  -D BUILD_opencv_python3=OFF \
  -D BUILD_opencv_python2=OFF \
  -D BUILD_SHARED_LIBS=ON \
  -D BUILD_opencv_apps=OFF \
  -D WITH_FFMPEG=ON 
  
make -j8
make install
cd ..
```
编译完opencv后，你就可以按照上述的步骤，继续编译lite.ai.toolkit.

* Windows: 你可以参考[issue#6](https://github.com/DefTruth/lite.ai.toolkit/issues/6) ，讨论了常见的编译问题。
* Linux: 参考MacOS下的编译，替换Linux版本的依赖库即可。Linux下的发行版本将会在近期添加 ~ [issue#2](https://github.com/DefTruth/lite.ai.toolkit/issues/2)
* 令人开心的消息!!! : 🚀 你可以直接下载最新的**ONNXRuntime**官方构建的动态库，包含Windows, Linux, MacOS and Arm的版本!!! CPU和GPU的版本均可获得。不需要再从源码进行编译了，nice。可以从[v1.8.1](https://github.com/microsoft/onnxruntime/releases) 下载最新的动态库. 我目前在**Lite.Ai.ToolKit**中用的是1.7.0，你可以从[v1.7.0](https://github.com/microsoft/onnxruntime/releases/tag/v1.7.0) 下载, 但1.8.1应该也是可行的。对于**OpenCV**，请尝试从源码构建(Linux) 或者 直接从[OpenCV 4.5.3](https://github.com/opencv/opencv/releases) 下载官方编译好的动态库(Windows). 然后把头文件和依赖库放入上述的文件夹中.

* Windows GPU 兼容性: 详见[issue#10](https://github.com/DefTruth/lite.ai.toolkit/issues/10).
* Linux GPU 兼容性: 详见[issue#97](https://github.com/DefTruth/lite.ai.toolkit/issues/97).

</details>  

<details>
<summary>🔑️ 如何链接Lite.Ai.ToolKit动态库?</summary>  

* 你可参考以下的CMakeLists.txt设置来链接动态库.

```cmake
cmake_minimum_required(VERSION 3.17)
project(lite.ai.toolkit.demo)

set(CMAKE_CXX_STANDARD 11)

# setting up lite.ai.toolkit
set(LITE_AI_DIR ${CMAKE_SOURCE_DIR}/lite.ai.toolkit)
set(LITE_AI_INCLUDE_DIR ${LITE_AI_DIR}/include)
set(LITE_AI_LIBRARY_DIR ${LITE_AI_DIR}/lib)
include_directories(${LITE_AI_INCLUDE_DIR})
link_directories(${LITE_AI_LIBRARY_DIR})

set(OpenCV_LIBS
        opencv_highgui
        opencv_core
        opencv_imgcodecs
        opencv_imgproc
        opencv_video
        opencv_videoio
        )
# add your executable
set(EXECUTABLE_OUTPUT_PATH ${CMAKE_SOURCE_DIR}/examples/build)

add_executable(lite_rvm examples/test_lite_rvm.cpp)
target_link_libraries(lite_rvm
        lite.ai.toolkit
        onnxruntime
        MNN  # need, if built lite.ai.toolkit with ENABLE_MNN=ON,  default OFF
        ncnn # need, if built lite.ai.toolkit with ENABLE_NCNN=ON, default OFF 
        TNN  # need, if built lite.ai.toolkit with ENABLE_TNN=ON,  default OFF 
        ${OpenCV_LIBS})  # link lite.ai.toolkit & other libs.
```

```shell
cd ./build/lite.ai.toolkit/lib && otool -L liblite.ai.toolkit.0.0.1.dylib 
liblite.ai.toolkit.0.0.1.dylib:
        @rpath/liblite.ai.toolkit.0.0.1.dylib (compatibility version 0.0.1, current version 0.0.1)
        @rpath/libopencv_highgui.4.5.dylib (compatibility version 4.5.0, current version 4.5.2)
        @rpath/libonnxruntime.1.7.0.dylib (compatibility version 0.0.0, current version 1.7.0)
        ...
```


```shell
cd ../ && tree .
├── bin
├── include
│   ├── lite
│   │   ├── backend.h
│   │   ├── config.h
│   │   └── lite.h
│   └── ort
└── lib
    └── liblite.ai.toolkit.0.0.1.dylib
```
* 运行已经编译好的examples:
```shell
cd ./build/lite.ai.toolkit/bin && ls -lh | grep lite
-rwxr-xr-x  1 root  staff   301K Jun 26 23:10 liblite.ai.toolkit.0.0.1.dylib
...
-rwxr-xr-x  1 root  staff   196K Jun 26 23:10 lite_yolov4
-rwxr-xr-x  1 root  staff   196K Jun 26 23:10 lite_yolov5
...
```

```shell
./lite_yolov5
LITEORT_DEBUG LogId: ../../../hub/onnx/cv/yolov5s.onnx
=============== Input-Dims ==============
...
detected num_anchors: 25200
generate_bboxes num: 66
Default Version Detected Boxes Num: 5
```

为了链接`lite.ai.toolkit`动态库，你需要确保`OpenCV` and `onnxruntime`也被正确地链接。你可以在[CMakeLists.txt](https://github.com/DefTruth/RobustVideoMatting-ncnn-mnn-tnn-onnxruntime/blob/main/CMakeLists.txt) 中找到一个简单且完整的，关于如何正确地链接Lite.AI.ToolKit动态库的应用案例。

</details>


## 5. 模型下载  
<div id="lite.ai.toolkit-2"></div>
<div id="lite.ai.toolkit-Model-Zoo"></div>

**Lite.Ai.ToolKit** 目前包括 **[80+](https://github.com/DefTruth/lite.ai.toolkit/tree/main/docs/hub/lite.ai.toolkit.hub.onnx.md)** 流行的开源模型以及 **[500+](https://github.com/DefTruth/lite.ai.toolkit/tree/main/docs/hub/lite.ai.toolkit.hub.onnx.md)** 文件，大部分文件是我自己转换的。你可以通过**lite::cv::Type::Class** 语法进行调用，如 **[lite::cv::detection::YoloV5](#lite.ai.toolkit-object-detection)**。更多的细节见[Examples for Lite.Ai.ToolKit](#lite.ai.toolkit-Examples-for-Lite.AI.ToolKit)。注意，由于Google Driver(15G)的存储限制，我无法上传所有的模型文件，国内的小伙伴请使用百度云盘。

| File |                                Baidu Drive                                |                                             Google Drive                                             |                                                          Docker Hub                                                          |                                               Hub (Docs)                                               |  
|:----:|:-------------------------------------------------------------------------:|:----------------------------------------------------------------------------------------------------:|:----------------------------------------------------------------------------------------------------------------------------:|:------------------------------------------------------------------------------------------------------:|
| ONNX | [Baidu Drive](https://pan.baidu.com/s/1elUGcx7CZkkjEoYhTMwTRQ) code: 8gin | [Google Drive](https://drive.google.com/drive/folders/1p6uBcxGeyS1exc-T61vL8YRhwjYL4iD2?usp=sharing) | [ONNX Docker v0.1.22.01.08 (28G), v0.1.22.02.02 (400M)](https://hub.docker.com/r/qyjdefdocker/lite.ai.toolkit-onnx-hub/tags) | [ONNX Hub](https://github.com/DefTruth/lite.ai.toolkit/tree/main/docs/hub/lite.ai.toolkit.hub.onnx.md) |    
| MNN  | [Baidu Drive](https://pan.baidu.com/s/1KyO-bCYUv6qPq2M8BH_Okg) code: 9v63 |                                                  ❔                                                   |  [MNN Docker v0.1.22.01.08 (11G), v0.1.22.02.02 (213M)](https://hub.docker.com/r/qyjdefdocker/lite.ai.toolkit-mnn-hub/tags)  |  [MNN Hub](https://github.com/DefTruth/lite.ai.toolkit/tree/main/docs/hub/lite.ai.toolkit.hub.mnn.md)  |  
| NCNN | [Baidu Drive](https://pan.baidu.com/s/1hlnqyNsFbMseGFWscgVhgQ) code: sc7f |                                                  ❔                                                   | [NCNN Docker v0.1.22.01.08 (9G), v0.1.22.02.02 (197M)](https://hub.docker.com/r/qyjdefdocker/lite.ai.toolkit-ncnn-hub/tags)  | [NCNN Hub](https://github.com/DefTruth/lite.ai.toolkit/tree/main/docs/hub/lite.ai.toolkit.hub.ncnn.md) |
| TNN  | [Baidu Drive](https://pan.baidu.com/s/1lvM2YKyUbEc5HKVtqITpcw) code: 6o6k |                                                  ❔                                                   |  [TNN Docker v0.1.22.01.08 (11G), v0.1.22.02.02 (217M)](https://hub.docker.com/r/qyjdefdocker/lite.ai.toolkit-tnn-hub/tags)  |  [TNN Hub](https://github.com/DefTruth/lite.ai.toolkit/tree/main/docs/hub/lite.ai.toolkit.hub.tnn.md)  |

```shell
  docker pull qyjdefdocker/lite.ai.toolkit-onnx-hub:v0.1.22.01.08  # (28G)
  docker pull qyjdefdocker/lite.ai.toolkit-mnn-hub:v0.1.22.01.08   # (11G)
  docker pull qyjdefdocker/lite.ai.toolkit-ncnn-hub:v0.1.22.01.08  # (9G)
  docker pull qyjdefdocker/lite.ai.toolkit-tnn-hub:v0.1.22.01.08   # (11G)
  docker pull qyjdefdocker/lite.ai.toolkit-onnx-hub:v0.1.22.02.02  # (400M) + YOLO5Face
  docker pull qyjdefdocker/lite.ai.toolkit-mnn-hub:v0.1.22.02.02   # (213M) + YOLO5Face
  docker pull qyjdefdocker/lite.ai.toolkit-ncnn-hub:v0.1.22.02.02  # (197M) + YOLO5Face
  docker pull qyjdefdocker/lite.ai.toolkit-tnn-hub:v0.1.22.02.02   # (217M) + YOLO5Face
```
<details>
<summary> ❇️ 命名空间和Lite.Ai.ToolKit算法模块的对应关系 </summary>  

### 命名空间和Lite.Ai.ToolKit算法模块的对应关系

| Namepace                   | Details                                                                            |
|:---------------------------|:-----------------------------------------------------------------------------------|
| *lite::cv::detection*      | Object Detection. one-stage and anchor-free detectors, YoloV5, YoloV4, SSD, etc. ✅ |
| *lite::cv::classification* | Image Classification. DensNet, ShuffleNet, ResNet, IBNNet, GhostNet, etc. ✅        |
| *lite::cv::faceid*         | Face Recognition. ArcFace, CosFace, CurricularFace, etc. ❇️                        |
| *lite::cv::face*           | Face Analysis. *detect*, *align*, *pose*, *attr*, etc. ❇️                          |
| *lite::cv::face::detect*   | Face Detection. UltraFace, RetinaFace, FaceBoxes, PyramidBox, etc. ❇️              |
| *lite::cv::face::align*    | Face Alignment. PFLD(106), FaceLandmark1000(1000 landmarks), PRNet, etc. ❇️        |
| *lite::cv::face::pose*     | Head Pose Estimation.  FSANet, etc. ❇️                                             |
| *lite::cv::face::attr*     | Face Attributes. Emotion, Age, Gender. EmotionFerPlus, VGG16Age, etc. ❇️           |
| *lite::cv::segmentation*   | Object Segmentation. Such as FCN, DeepLabV3, etc.  ❇️ ️                            |
| *lite::cv::style*          | Style Transfer. Contains neural style transfer now, such as FastStyleTransfer.  ⚠️ |
| *lite::cv::matting*        | Image Matting. Object and Human matting.   ❇️ ️                                    |
| *lite::cv::colorization*   | Colorization. Make Gray image become RGB. ⚠️                                       |
| *lite::cv::resolution*     | Super Resolution.  ⚠️                                                              |


### Lite.Ai.ToolKit的类与权重文件对应关系说明

Lite.AI.ToolKit的类与权重文件对应关系说明，可以在[lite.ai.toolkit.hub.onnx.md](https://github.com/DefTruth/lite.ai.toolkit/tree/main/docs/hub/lite.ai.toolkit.hub.onnx.md) 中找到。比如, *lite::cv::detection::YoloV5* 和 *lite::cv::detection::YoloX* 的权重文件为：


|             Class             | Pretrained ONNX Files |                 Rename or Converted From (Repo)                  | Size  |
|:-----------------------------:|:---------------------:|:----------------------------------------------------------------:|:-----:|
| *lite::cv::detection::YoloV5* |     yolov5l.onnx      |    [yolov5](https://github.com/ultralytics/yolov5) (🔥🔥💥↑)     | 188Mb |
| *lite::cv::detection::YoloV5* |     yolov5m.onnx      |    [yolov5](https://github.com/ultralytics/yolov5) (🔥🔥💥↑)     | 85Mb  |
| *lite::cv::detection::YoloV5* |     yolov5s.onnx      |    [yolov5](https://github.com/ultralytics/yolov5) (🔥🔥💥↑)     | 29Mb  |
| *lite::cv::detection::YoloV5* |     yolov5x.onnx      |    [yolov5](https://github.com/ultralytics/yolov5) (🔥🔥💥↑)     | 351Mb |
| *lite::cv::detection::YoloX*  |     yolox_x.onnx      | [YOLOX](https://github.com/Megvii-BaseDetection/YOLOX) (🔥🔥!!↑) | 378Mb |
| *lite::cv::detection::YoloX*  |     yolox_l.onnx      | [YOLOX](https://github.com/Megvii-BaseDetection/YOLOX) (🔥🔥!!↑) | 207Mb |
| *lite::cv::detection::YoloX*  |     yolox_m.onnx      | [YOLOX](https://github.com/Megvii-BaseDetection/YOLOX) (🔥🔥!!↑) | 97Mb  |
| *lite::cv::detection::YoloX*  |     yolox_s.onnx      | [YOLOX](https://github.com/Megvii-BaseDetection/YOLOX) (🔥🔥!!↑) | 34Mb  |
| *lite::cv::detection::YoloX*  |    yolox_tiny.onnx    | [YOLOX](https://github.com/Megvii-BaseDetection/YOLOX) (🔥🔥!!↑) | 19Mb  |
| *lite::cv::detection::YoloX*  |    yolox_nano.onnx    | [YOLOX](https://github.com/Megvii-BaseDetection/YOLOX) (🔥🔥!!↑) | 3.5Mb |

这意味着，你可以通过Lite.Ai.ToolKit中的同一个类，根据你的使用情况，加载任意一个`yolov5*.onnx`或`yolox_*.onnx`，如 *YoloV5*, *YoloX*等.

```c++
auto *yolov5 = new lite::cv::detection::YoloV5("yolov5x.onnx");  // for server
auto *yolov5 = new lite::cv::detection::YoloV5("yolov5l.onnx"); 
auto *yolov5 = new lite::cv::detection::YoloV5("yolov5m.onnx");  
auto *yolov5 = new lite::cv::detection::YoloV5("yolov5s.onnx");  // for mobile device 
auto *yolox = new lite::cv::detection::YoloX("yolox_x.onnx");  
auto *yolox = new lite::cv::detection::YoloX("yolox_l.onnx");  
auto *yolox = new lite::cv::detection::YoloX("yolox_m.onnx");  
auto *yolox = new lite::cv::detection::YoloX("yolox_s.onnx");  
auto *yolox = new lite::cv::detection::YoloX("yolox_tiny.onnx");  
auto *yolox = new lite::cv::detection::YoloX("yolox_nano.onnx");  // 3.5Mb only !
```

</details>  

<details>
<summary> 🔑️ 如何从通过Docker Hub下载Model Zoo？</summary>    

* Firstly, pull the image from docker hub.
  ```shell
  docker pull qyjdefdocker/lite.ai.toolkit-mnn-hub:v0.1.22.01.08 # (11G)
  docker pull qyjdefdocker/lite.ai.toolkit-ncnn-hub:v0.1.22.01.08 # (9G)
  docker pull qyjdefdocker/lite.ai.toolkit-tnn-hub:v0.1.22.01.08 # (11G)
  docker pull qyjdefdocker/lite.ai.toolkit-onnx-hub:v0.1.22.01.08 # (28G)
  ```
* Secondly, run the container with local `share` dir using `docker run -idt xxx`. A minimum example will show you as follows.
  * make a `share` dir in your local device.
  ```shell
  mkdir share # any name is ok.
  ```
  * write `run_mnn_docker_hub.sh` script like:
  ```shell
  #!/bin/bash  
  PORT1=6072
  PORT2=6084
  SERVICE_DIR=/Users/xxx/Desktop/your-path-to/share
  CONRAINER_DIR=/home/hub/share
  CONRAINER_NAME=mnn_docker_hub_d
  
  docker run -idt -p ${PORT2}:${PORT1} -v ${SERVICE_DIR}:${CONRAINER_DIR} --shm-size=16gb --name ${CONRAINER_NAME} qyjdefdocker/lite.ai.toolkit-mnn-hub:v0.1.22.01.08

  ```
* Finally, copy the model weights from `/home/hub/mnn/cv` to your local `share` dir.
  ```shell
  # activate mnn docker.
  sh ./run_mnn_docker_hub.sh
  docker exec -it mnn_docker_hub_d /bin/bash
  # copy the models to the share dir.
  cd /home/hub 
  cp -rf mnn/cv share/
  ```

</details>

### Model Hubs
lite.ai.toolkit提供大量的预训练模型的ONNX文件. 同时, 更多的模型权重文件详见 [Model Zoo](#lite.ai.toolkit-Model-Zoo) and [ONNX Hub](https://github.com/DefTruth/lite.ai.toolkit/tree/main/docs/hub/lite.ai.toolkit.hub.onnx.md), [MNN Hub](https://github.com/DefTruth/lite.ai.toolkit/tree/main/docs/hub/lite.ai.toolkit.hub.mnn.md), [TNN Hub](https://github.com/DefTruth/lite.ai.toolkit/tree/main/docs/hub/lite.ai.toolkit.hub.tnn.md), [NCNN Hub](https://github.com/DefTruth/lite.ai.toolkit/tree/main/docs/hub/lite.ai.toolkit.hub.ncnn.md).
<details>
<summary> 🔑️ ONNX Model Hub </summary>
#### Object Detection.

<div id="lite.ai.toolkit.hub.onnx-object-detection"></div>

|                     Class                      |         Pretrained ONNX Files          |                          Rename or Converted From (Repo)                          | Size  |
|:----------------------------------------------:|:--------------------------------------:|:---------------------------------------------------------------------------------:|:-----:|
|         *lite::cv::detection::YoloV5*          |              yolov5l.onnx              |                  [yolov5](https://github.com/ultralytics/yolov5)                  | 188Mb |
|         *lite::cv::detection::YoloV5*          |              yolov5m.onnx              |                  [yolov5](https://github.com/ultralytics/yolov5)                  | 85Mb  |
|         *lite::cv::detection::YoloV5*          |              yolov5s.onnx              |                  [yolov5](https://github.com/ultralytics/yolov5)                  | 29Mb  |
|         *lite::cv::detection::YoloV5*          |              yolov5x.onnx              |                  [yolov5](https://github.com/ultralytics/yolov5)                  | 351Mb |
|          *lite::cv::detection::YoloX*          |              yolox_x.onnx              |              [YOLOX](https://github.com/Megvii-BaseDetection/YOLOX)               | 378Mb |
|          *lite::cv::detection::YoloX*          |              yolox_l.onnx              |              [YOLOX](https://github.com/Megvii-BaseDetection/YOLOX)               | 207Mb |
|          *lite::cv::detection::YoloX*          |              yolox_m.onnx              |              [YOLOX](https://github.com/Megvii-BaseDetection/YOLOX)               | 97Mb  |
|          *lite::cv::detection::YoloX*          |              yolox_s.onnx              |              [YOLOX](https://github.com/Megvii-BaseDetection/YOLOX)               | 34Mb  |
|          *lite::cv::detection::YoloX*          |            yolox_tiny.onnx             |              [YOLOX](https://github.com/Megvii-BaseDetection/YOLOX)               | 19Mb  |
|          *lite::cv::detection::YoloX*          |            yolox_nano.onnx             |              [YOLOX](https://github.com/Megvii-BaseDetection/YOLOX)               | 3.5Mb |
|         *lite::cv::detection::YoloV3*          |             yolov3-10.onnx             |                   [onnx-models](https://github.com/onnx/models)                   | 236Mb |
|       *lite::cv::detection::TinyYoloV3*        |          tiny-yolov3-11.onnx           |                   [onnx-models](https://github.com/onnx/models)                   | 33Mb  |
|         *lite::cv::detection::YoloV4*          |    voc-mobilenetv2-yolov4-640.onnx     |             [YOLOv4...](https://github.com/argusswift/YOLOv4-pytorch)             | 176Mb |
|         *lite::cv::detection::YoloV4*          |    voc-mobilenetv2-yolov4-416.onnx     |             [YOLOv4...](https://github.com/argusswift/YOLOv4-pytorch)             | 176Mb |
|           *lite::cv::detection::SSD*           |              ssd-10.onnx               |                   [onnx-models](https://github.com/onnx/models)                   | 76Mb  |
|          *lite::cv::detection::YoloR*          |        yolor-d6-1280-1280.onnx         |                   [yolor](https://github.com/WongKinYiu/yolor)                    | 667Mb |
|          *lite::cv::detection::YoloR*          |         yolor-d6-640-640.onnx          |                   [yolor](https://github.com/WongKinYiu/yolor)                    | 601Mb |
|          *lite::cv::detection::YoloR*          |         yolor-d6-320-320.onnx          |                   [yolor](https://github.com/WongKinYiu/yolor)                    | 584Mb |
|          *lite::cv::detection::YoloR*          |        yolor-e6-1280-1280.onnx         |                   [yolor](https://github.com/WongKinYiu/yolor)                    | 530Mb |
|          *lite::cv::detection::YoloR*          |         yolor-e6-640-640.onnx          |                   [yolor](https://github.com/WongKinYiu/yolor)                    | 464Mb |
|          *lite::cv::detection::YoloR*          |         yolor-e6-320-320.onnx          |                   [yolor](https://github.com/WongKinYiu/yolor)                    | 448Mb |
|          *lite::cv::detection::YoloR*          |        yolor-p6-1280-1280.onnx         |                   [yolor](https://github.com/WongKinYiu/yolor)                    | 214Mb |
|          *lite::cv::detection::YoloR*          |         yolor-p6-640-640.onnx          |                   [yolor](https://github.com/WongKinYiu/yolor)                    | 160Mb |
|          *lite::cv::detection::YoloR*          |         yolor-p6-320-320.onnx          |                   [yolor](https://github.com/WongKinYiu/yolor)                    | 147Mb |
|          *lite::cv::detection::YoloR*          |        yolor-w6-1280-1280.onnx         |                   [yolor](https://github.com/WongKinYiu/yolor)                    | 382Mb |
|          *lite::cv::detection::YoloR*          |         yolor-w6-640-640.onnx          |                   [yolor](https://github.com/WongKinYiu/yolor)                    | 324Mb |
|          *lite::cv::detection::YoloR*          |         yolor-w6-320-320.onnx          |                   [yolor](https://github.com/WongKinYiu/yolor)                    | 309Mb |
|          *lite::cv::detection::YoloR*          |     yolor-ssss-s2d-1280-1280.onnx      |                   [yolor](https://github.com/WongKinYiu/yolor)                    | 90Mb  |
|          *lite::cv::detection::YoloR*          |      yolor-ssss-s2d-640-640.onnx       |                   [yolor](https://github.com/WongKinYiu/yolor)                    | 49Mb  |
|          *lite::cv::detection::YoloR*          |      yolor-ssss-s2d-320-320.onnx       |                   [yolor](https://github.com/WongKinYiu/yolor)                    | 39Mb  |
|      *lite::cv::detection::TinyYoloV4VOC*      |      yolov4_tiny_weights_voc.onnx      |       [yolov4-tiny...](https://github.com/bubbliiiing/yolov4-tiny-pytorch)        | 23Mb  |
|      *lite::cv::detection::TinyYoloV4VOC*      |    yolov4_tiny_weights_voc_SE.onnx     |       [yolov4-tiny...](https://github.com/bubbliiiing/yolov4-tiny-pytorch)        | 23Mb  |
|      *lite::cv::detection::TinyYoloV4VOC*      |   yolov4_tiny_weights_voc_CBAM.onnx    |       [yolov4-tiny...](https://github.com/bubbliiiing/yolov4-tiny-pytorch)        | 23Mb  |
|      *lite::cv::detection::TinyYoloV4VOC*      |    yolov4_tiny_weights_voc_ECA.onnx    |       [yolov4-tiny...](https://github.com/bubbliiiing/yolov4-tiny-pytorch)        | 23Mb  |
|     *lite::cv::detection::TinyYoloV4COCO*      |     yolov4_tiny_weights_coco.onnx      |       [yolov4-tiny...](https://github.com/bubbliiiing/yolov4-tiny-pytorch)        | 23Mb  |
|      *lite::cv::detection::ScaledYoloV4*       | ScaledYoloV4_yolov4-p5-1280-1280.onnx  |            [ScaledYOLOv4](https://github.com/WongKinYiu/ScaledYOLOv4)             | 270Mb |
|      *lite::cv::detection::ScaledYoloV4*       |  ScaledYoloV4_yolov4-p5-640-640.onnx   |            [ScaledYOLOv4](https://github.com/WongKinYiu/ScaledYOLOv4)             | 270Mb |
|      *lite::cv::detection::ScaledYoloV4*       |  ScaledYoloV4_yolov4-p5-320-320.onnx   |            [ScaledYOLOv4](https://github.com/WongKinYiu/ScaledYOLOv4)             | 270Mb |
|      *lite::cv::detection::ScaledYoloV4*       | ScaledYoloV4_yolov4-p6-1280-1280.onnx  |            [ScaledYOLOv4](https://github.com/WongKinYiu/ScaledYOLOv4)             | 487Mb |
|      *lite::cv::detection::ScaledYoloV4*       |  ScaledYoloV4_yolov4-p6-640-640.onnx   |            [ScaledYOLOv4](https://github.com/WongKinYiu/ScaledYOLOv4)             | 487Mb |
|      *lite::cv::detection::ScaledYoloV4*       |  ScaledYoloV4_yolov4-p6-320-320.onnx   |            [ScaledYOLOv4](https://github.com/WongKinYiu/ScaledYOLOv4)             | 487Mb |
|      *lite::cv::detection::ScaledYoloV4*       | ScaledYoloV4_yolov4-p7-1280-1280.onnx  |            [ScaledYOLOv4](https://github.com/WongKinYiu/ScaledYOLOv4)             | 1.1Gb |
|      *lite::cv::detection::ScaledYoloV4*       |  ScaledYoloV4_yolov4-p7-640-640.onnx   |            [ScaledYOLOv4](https://github.com/WongKinYiu/ScaledYOLOv4)             | 1.1Gb |
|      *lite::cv::detection::ScaledYoloV4*       | ScaledYoloV4_yolov4-p5_-1280-1280.onnx |            [ScaledYOLOv4](https://github.com/WongKinYiu/ScaledYOLOv4)             | 270Mb |
|      *lite::cv::detection::ScaledYoloV4*       |  ScaledYoloV4_yolov4-p5_-640-640.onnx  |            [ScaledYOLOv4](https://github.com/WongKinYiu/ScaledYOLOv4)             | 270Mb |
|      *lite::cv::detection::ScaledYoloV4*       |  ScaledYoloV4_yolov4-p5_-320-320.onnx  |            [ScaledYOLOv4](https://github.com/WongKinYiu/ScaledYOLOv4)             | 270Mb |
|      *lite::cv::detection::ScaledYoloV4*       | ScaledYoloV4_yolov4-p6_-1280-1280.onnx |            [ScaledYOLOv4](https://github.com/WongKinYiu/ScaledYOLOv4)             | 487Mb |
|      *lite::cv::detection::ScaledYoloV4*       |  ScaledYoloV4_yolov4-p6_-640-640.onnx  |            [ScaledYOLOv4](https://github.com/WongKinYiu/ScaledYOLOv4)             | 487Mb |
|      *lite::cv::detection::ScaledYoloV4*       |  ScaledYoloV4_yolov4-p6_-320-320.onnx  |            [ScaledYOLOv4](https://github.com/WongKinYiu/ScaledYOLOv4)             | 487Mb |
|      *lite::cv::detection::EfficientDet*       |          efficientdet-d0.onnx          | [...EfficientDet...](https://github.com/zylo117/Yet-Another-EfficientDet-Pytorch) | 15Mb  |
|      *lite::cv::detection::EfficientDet*       |          efficientdet-d1.onnx          | [...EfficientDet...](https://github.com/zylo117/Yet-Another-EfficientDet-Pytorch) | 26Mb  |
|      *lite::cv::detection::EfficientDet*       |          efficientdet-d2.onnx          | [...EfficientDet...](https://github.com/zylo117/Yet-Another-EfficientDet-Pytorch) | 32Mb  |
|      *lite::cv::detection::EfficientDet*       |          efficientdet-d3.onnx          | [...EfficientDet...](https://github.com/zylo117/Yet-Another-EfficientDet-Pytorch) | 49Mb  |
|      *lite::cv::detection::EfficientDet*       |          efficientdet-d4.onnx          | [...EfficientDet...](https://github.com/zylo117/Yet-Another-EfficientDet-Pytorch) | 85Mb  |
|      *lite::cv::detection::EfficientDet*       |          efficientdet-d5.onnx          | [...EfficientDet...](https://github.com/zylo117/Yet-Another-EfficientDet-Pytorch) | 138Mb |
|      *lite::cv::detection::EfficientDet*       |          efficientdet-d6.onnx          | [...EfficientDet...](https://github.com/zylo117/Yet-Another-EfficientDet-Pytorch) | 220Mb |
|     *lite::cv::detection::EfficientDetD7*      |          efficientdet-d7.onnx          | [...EfficientDet...](https://github.com/zylo117/Yet-Another-EfficientDet-Pytorch) | 220Mb |
|     *lite::cv::detection::EfficientDetD8*      |          efficientdet-d8.onnx          | [...EfficientDet...](https://github.com/zylo117/Yet-Another-EfficientDet-Pytorch) | 322Mb |
|          *lite::cv::detection::YOLOP*          |          yolop-1280-1280.onnx          |                     [YOLOP](https://github.com/hustvl/YOLOP)                      | 30Mb  |
|          *lite::cv::detection::YOLOP*          |           yolop-640-640.onnx           |                     [YOLOP](https://github.com/hustvl/YOLOP)                      | 30Mb  |
|          *lite::cv::detection::YOLOP*          |           yolop-320-320.onnx           |                     [YOLOP](https://github.com/hustvl/YOLOP)                      | 30Mb  |
|         *lite::cv::detection::NanoDet*         |          nanodet_m_0.5x.onnx           |                  [nanodet](https://github.com/RangiLyu/nanodet)                   | 1.1Mb |
|         *lite::cv::detection::NanoDet*         |             nanodet_m.onnx             |                  [nanodet](https://github.com/RangiLyu/nanodet)                   | 3.6Mb |
|         *lite::cv::detection::NanoDet*         |          nanodet_m_1.5x.onnx           |                  [nanodet](https://github.com/RangiLyu/nanodet)                   | 7.9Mb |
|         *lite::cv::detection::NanoDet*         |        nanodet_m_1.5x_416.onnx         |                  [nanodet](https://github.com/RangiLyu/nanodet)                   | 7.9Mb |
|         *lite::cv::detection::NanoDet*         |           nanodet_m_416.onnx           |                  [nanodet](https://github.com/RangiLyu/nanodet)                   | 3.6Mb |
|         *lite::cv::detection::NanoDet*         |             nanodet_g.onnx             |                  [nanodet](https://github.com/RangiLyu/nanodet)                   | 14Mb  |
|         *lite::cv::detection::NanoDet*         |             nanodet_t.onnx             |                  [nanodet](https://github.com/RangiLyu/nanodet)                   | 5.1Mb |
|         *lite::cv::detection::NanoDet*         |       nanodet-RepVGG-A0_416.onnx       |                  [nanodet](https://github.com/RangiLyu/nanodet)                   | 26Mb  |
| *lite::cv::detection::NanoDetEfficientNetLite* |  nanodet-EfficientNet-Lite0_320.onnx   |                  [nanodet](https://github.com/RangiLyu/nanodet)                   | 12Mb  |
| *lite::cv::detection::NanoDetEfficientNetLite* |  nanodet-EfficientNet-Lite1_416.onnx   |                  [nanodet](https://github.com/RangiLyu/nanodet)                   | 15Mb  |
| *lite::cv::detection::NanoDetEfficientNetLite* |  nanodet-EfficientNet-Lite2_512.onnx   |                  [nanodet](https://github.com/RangiLyu/nanodet)                   | 18Mb  |
|      *lite::cv::detection::YoloX_V_0_1_1*      |          yolox_x_v0.1.1.onnx           |              [YOLOX](https://github.com/Megvii-BaseDetection/YOLOX)               | 378Mb |
|      *lite::cv::detection::YoloX_V_0_1_1*      |          yolox_l_v0.1.1.onnx           |              [YOLOX](https://github.com/Megvii-BaseDetection/YOLOX)               | 207Mb |
|      *lite::cv::detection::YoloX_V_0_1_1*      |          yolox_m_v0.1.1.onnx           |              [YOLOX](https://github.com/Megvii-BaseDetection/YOLOX)               | 97Mb  |
|      *lite::cv::detection::YoloX_V_0_1_1*      |          yolox_s_v0.1.1.onnx           |              [YOLOX](https://github.com/Megvii-BaseDetection/YOLOX)               | 34Mb  |
|      *lite::cv::detection::YoloX_V_0_1_1*      |         yolox_tiny_v0.1.1.onnx         |              [YOLOX](https://github.com/Megvii-BaseDetection/YOLOX)               | 19Mb  |
|      *lite::cv::detection::YoloX_V_0_1_1*      |         yolox_nano_v0.1.1.onnx         |              [YOLOX](https://github.com/Megvii-BaseDetection/YOLOX)               | 3.5Mb |
|      *lite::cv::detection::YoloV5_V_6_0*       |       yolov5l.640-640.v.6.0.onnx       |                  [yolov5](https://github.com/ultralytics/yolov5)                  | 178Mb |
|      *lite::cv::detection::YoloV5_V_6_0*       |       yolov5m.640-640.v.6.0.onnx       |                  [yolov5](https://github.com/ultralytics/yolov5)                  | 81Mb  |
|      *lite::cv::detection::YoloV5_V_6_0*       |       yolov5s.640-640.v.6.0.onnx       |                  [yolov5](https://github.com/ultralytics/yolov5)                  | 28Mb  |
|      *lite::cv::detection::YoloV5_V_6_0*       |       yolov5x.640-640.v.6.0.onnx       |                  [yolov5](https://github.com/ultralytics/yolov5)                  | 331Mb |
|      *lite::cv::detection::YoloV5_V_6_0*       |       yolov5n.640-640.v.6.0.onnx       |                  [yolov5](https://github.com/ultralytics/yolov5)                  | 7.5Mb |
|      *lite::cv::detection::YoloV5_V_6_0*       |      yolov5l6.640-640.v.6.0.onnx       |                  [yolov5](https://github.com/ultralytics/yolov5)                  | 294Mb |
|      *lite::cv::detection::YoloV5_V_6_0*       |      yolov5m6.640-640.v.6.0.onnx       |                  [yolov5](https://github.com/ultralytics/yolov5)                  | 128Mb |
|      *lite::cv::detection::YoloV5_V_6_0*       |      yolov5s6.640-640.v.6.0.onnx       |                  [yolov5](https://github.com/ultralytics/yolov5)                  | 50Mb  |
|      *lite::cv::detection::YoloV5_V_6_0*       |      yolov5x6.640-640.v.6.0.onnx       |                  [yolov5](https://github.com/ultralytics/yolov5)                  | 538Mb |
|      *lite::cv::detection::YoloV5_V_6_0*       |      yolov5n6.640-640.v.6.0.onnx       |                  [yolov5](https://github.com/ultralytics/yolov5)                  | 14Mb  |
|      *lite::cv::detection::YoloV5_V_6_0*       |     yolov5l6.1280-1280.v.6.0.onnx      |                  [yolov5](https://github.com/ultralytics/yolov5)                  | 294Mb |
|      *lite::cv::detection::YoloV5_V_6_0*       |     yolov5m6.1280-1280.v.6.0.onnx      |                  [yolov5](https://github.com/ultralytics/yolov5)                  | 128Mb |
|      *lite::cv::detection::YoloV5_V_6_0*       |     yolov5s6.1280-1280.v.6.0.onnx      |                  [yolov5](https://github.com/ultralytics/yolov5)                  | 50Mb  |
|      *lite::cv::detection::YoloV5_V_6_0*       |     yolov5x6.1280-1280.v.6.0.onnx      |                  [yolov5](https://github.com/ultralytics/yolov5)                  | 538Mb |
|      *lite::cv::detection::YoloV5_V_6_0*       |     yolov5n6.1280-1280.v.6.0.onnx      |                  [yolov5](https://github.com/ultralytics/yolov5)                  | 14Mb  |
|       *lite::cv::detection::NanoDetPlus*       |        nanodet-plus-m_320.onnx         |                  [nanodet](https://github.com/RangiLyu/nanodet)                   | 4.5Mb |
|       *lite::cv::detection::NanoDetPlus*       |        nanodet-plus-m_416.onnx         |                  [nanodet](https://github.com/RangiLyu/nanodet)                   | 4.5Mb |
|       *lite::cv::detection::NanoDetPlus*       |      nanodet-plus-m-1.5x_320.onnx      |                  [nanodet](https://github.com/RangiLyu/nanodet)                   | 9.4Mb |
|       *lite::cv::detection::NanoDetPlus*       |      nanodet-plus-m-1.5x_416.onnx      |                  [nanodet](https://github.com/RangiLyu/nanodet)                   | 9.4Mb |
|        *lite::cv::detection::InsectDet*        |     quarrying_insect_detector.onnx     |           [InsectID](https://github.com/quarrying/quarrying-insect-id)            | 22Mb  |

#### Classification.

<div id="lite.ai.toolkit.hub.onnx-classification"></div>


|                    Class                     |      Pretrained ONNX Files       |               Rename or Converted From (Repo)                | Size  |
|:--------------------------------------------:|:--------------------------------:|:------------------------------------------------------------:|:-----:|
| *lite::cv::classification:EfficientNetLite4* |    efficientnet-lite4-11.onnx    |        [onnx-models](https://github.com/onnx/models)         | 49Mb  |
|   *lite::cv::classification::ShuffleNetV2*   |      shufflenet-v2-10.onnx       |        [onnx-models](https://github.com/onnx/models)         | 8.7Mb |
|   *lite::cv::classification::DenseNet121*    |         densenet121.onnx         |       [torchvision](https://github.com/pytorch/vision)       | 30Mb  |
|     *lite::cv::classification::GhostNet*     |          ghostnet.onnx           |       [torchvision](https://github.com/pytorch/vision)       | 20Mb  |
|     *lite::cv::classification::HdrDNet*      |           hardnet.onnx           |       [torchvision](https://github.com/pytorch/vision)       | 13Mb  |
|      *lite::cv::classification::IBNNet*      |          ibnnet18.onnx           |       [torchvision](https://github.com/pytorch/vision)       | 97Mb  |
|   *lite::cv::classification::MobileNetV2*    |         mobilenetv2.onnx         |       [torchvision](https://github.com/pytorch/vision)       | 13Mb  |
|      *lite::cv::classification::ResNet*      |          resnet18.onnx           |       [torchvision](https://github.com/pytorch/vision)       | 44Mb  |
|     *lite::cv::classification::ResNeXt*      |           resnext.onnx           |       [torchvision](https://github.com/pytorch/vision)       | 95Mb  |
|     *lite::cv::classification::InsectID*     | quarrying_insect_identifier.onnx | [InsectID](https://github.com/quarrying/quarrying-insect-id) | 27Mb  |
|      *lite::cv::classification:PlantID*      |   quarrying_plantid_model.onnx   |  [PlantID](https://github.com/quarrying/quarrying-plant-id)  | 30Mb  |


#### Face Detection.

<div id="lite.ai.toolkit.hub.onnx-face-detection"></div>  

|                 Class                 |            Pretrained ONNX Files            |                             Rename or Converted From (Repo)                             |  Size  |
|:-------------------------------------:|:-------------------------------------------:|:---------------------------------------------------------------------------------------:|:------:|
|  *lite::cv::face::detect::UltraFace*  |           ultraface-rfb-640.onnx            | [Ultra-Light...](https://github.com/Linzaer/Ultra-Light-Fast-Generic-Face-Detector-1MB) | 1.5Mb  |
|  *lite::cv::face::detect::UltraFace*  |           ultraface-rfb-320.onnx            | [Ultra-Light...](https://github.com/Linzaer/Ultra-Light-Fast-Generic-Face-Detector-1MB) | 1.2Mb  |
| *lite::cv::face::detect::RetinaFace*  |      Pytorch_RetinaFace_resnet50.onnx       |             [...Retinaface](https://github.com/biubug6/Pytorch_Retinaface)              | 104Mb  |
| *lite::cv::face::detect::RetinaFace*  |  Pytorch_RetinaFace_resnet50-640-640.onnx   |             [...Retinaface](https://github.com/biubug6/Pytorch_Retinaface)              | 104Mb  |
| *lite::cv::face::detect::RetinaFace*  |  Pytorch_RetinaFace_resnet50-320-320.onnx   |             [...Retinaface](https://github.com/biubug6/Pytorch_Retinaface)              | 104Mb  |
| *lite::cv::face::detect::RetinaFace*  |  Pytorch_RetinaFace_resnet50-720-1080.onnx  |             [...Retinaface](https://github.com/biubug6/Pytorch_Retinaface)              | 104Mb  |
| *lite::cv::face::detect::RetinaFace*  |     Pytorch_RetinaFace_mobile0.25.onnx      |             [...Retinaface](https://github.com/biubug6/Pytorch_Retinaface)              | 1.6Mb  |
| *lite::cv::face::detect::RetinaFace*  | Pytorch_RetinaFace_mobile0.25-640-640.onnx  |             [...Retinaface](https://github.com/biubug6/Pytorch_Retinaface)              | 1.6Mb  |
| *lite::cv::face::detect::RetinaFace*  | Pytorch_RetinaFace_mobile0.25-320-320.onnx  |             [...Retinaface](https://github.com/biubug6/Pytorch_Retinaface)              | 1.6Mb  |
| *lite::cv::face::detect::RetinaFace*  | Pytorch_RetinaFace_mobile0.25-720-1080.onnx |             [...Retinaface](https://github.com/biubug6/Pytorch_Retinaface)              | 1.6Mb  |
|  *lite::cv::face::detect::FaceBoxes*  |               FaceBoxes.onnx                |                [FaceBoxes](https://github.com/zisianw/FaceBoxes.PyTorch)                | 3.8Mb  |
|  *lite::cv::face::detect::FaceBoxes*  |           FaceBoxes-640-640.onnx            |                [FaceBoxes](https://github.com/zisianw/FaceBoxes.PyTorch)                | 3.8Mb  |
|  *lite::cv::face::detect::FaceBoxes*  |           FaceBoxes-320-320.onnx            |                [FaceBoxes](https://github.com/zisianw/FaceBoxes.PyTorch)                | 3.8Mb  |
|  *lite::cv::face::detect::FaceBoxes*  |           FaceBoxes-720-1080.onnx           |                [FaceBoxes](https://github.com/zisianw/FaceBoxes.PyTorch)                | 3.8Mb  |
|    *lite::cv::face::detect::SCRFD*    |        scrfd_500m_shape160x160.onnx         |     [SCRFD](https://github.com/deepinsight/insightface/blob/master/detection/scrfd)     | 2.5Mb  |
|    *lite::cv::face::detect::SCRFD*    |        scrfd_500m_shape320x320.onnx         |     [SCRFD](https://github.com/deepinsight/insightface/blob/master/detection/scrfd)     | 2.5Mb  |
|    *lite::cv::face::detect::SCRFD*    |        scrfd_500m_shape640x640.onnx         |     [SCRFD](https://github.com/deepinsight/insightface/blob/master/detection/scrfd)     | 2.5Mb  |
|    *lite::cv::face::detect::SCRFD*    |     scrfd_500m_bnkps_shape160x160.onnx      |     [SCRFD](https://github.com/deepinsight/insightface/blob/master/detection/scrfd)     | 2.5Mb  |  
|    *lite::cv::face::detect::SCRFD*    |     scrfd_500m_bnkps_shape320x320.onnx      |     [SCRFD](https://github.com/deepinsight/insightface/blob/master/detection/scrfd)     | 2.5Mb  |  
|    *lite::cv::face::detect::SCRFD*    |     scrfd_500m_bnkps_shape640x640.onnx      |     [SCRFD](https://github.com/deepinsight/insightface/blob/master/detection/scrfd)     | 2.5Mb  |  
|    *lite::cv::face::detect::SCRFD*    |         scrfd_1g_shape160x160.onnx          |     [SCRFD](https://github.com/deepinsight/insightface/blob/master/detection/scrfd)     | 2.7Mb  |
|    *lite::cv::face::detect::SCRFD*    |         scrfd_1g_shape320x320.onnx          |     [SCRFD](https://github.com/deepinsight/insightface/blob/master/detection/scrfd)     | 2.7Mb  |
|    *lite::cv::face::detect::SCRFD*    |         scrfd_1g_shape640x640.onnx          |     [SCRFD](https://github.com/deepinsight/insightface/blob/master/detection/scrfd)     | 2.7Mb  |
|    *lite::cv::face::detect::SCRFD*    |        scrfd_2.5g_shape160x160.onnx         |     [SCRFD](https://github.com/deepinsight/insightface/blob/master/detection/scrfd)     | 3.3Mb  |
|    *lite::cv::face::detect::SCRFD*    |        scrfd_2.5g_shape320x320.onnx         |     [SCRFD](https://github.com/deepinsight/insightface/blob/master/detection/scrfd)     | 3.3Mb  |
|    *lite::cv::face::detect::SCRFD*    |        scrfd_2.5g_shape640x640.onnx         |     [SCRFD](https://github.com/deepinsight/insightface/blob/master/detection/scrfd)     | 3.3Mb  |
|    *lite::cv::face::detect::SCRFD*    |     scrfd_2.5g_bnkps_shape160x160.onnx      |     [SCRFD](https://github.com/deepinsight/insightface/blob/master/detection/scrfd)     | 3.3Mb  |  
|    *lite::cv::face::detect::SCRFD*    |     scrfd_2.5g_bnkps_shape320x320.onnx      |     [SCRFD](https://github.com/deepinsight/insightface/blob/master/detection/scrfd)     | 3.3Mb  |  
|    *lite::cv::face::detect::SCRFD*    |     scrfd_2.5g_bnkps_shape640x640.onnx      |     [SCRFD](https://github.com/deepinsight/insightface/blob/master/detection/scrfd)     | 3.3Mb  |  
|    *lite::cv::face::detect::SCRFD*    |         scrfd_10g_shape640x640.onnx         |     [SCRFD](https://github.com/deepinsight/insightface/blob/master/detection/scrfd)     | 16.9Mb |
|    *lite::cv::face::detect::SCRFD*    |        scrfd_10g_shape1280x1280.onnx        |     [SCRFD](https://github.com/deepinsight/insightface/blob/master/detection/scrfd)     | 16.9Mb |
|    *lite::cv::face::detect::SCRFD*    |      scrfd_10g_bnkps_shape640x640.onnx      |     [SCRFD](https://github.com/deepinsight/insightface/blob/master/detection/scrfd)     | 16.9Mb |  
|    *lite::cv::face::detect::SCRFD*    |     scrfd_10g_bnkps_shape1280x1280.onnx     |     [SCRFD](https://github.com/deepinsight/insightface/blob/master/detection/scrfd)     | 16.9Mb |  
|  *lite::cv::face::detect::YOLO5Face*  |      yolov5face-blazeface-640x640.onnx      |                 [YOLO5Face](https://github.com/deepcam-cn/yolov5-face)                  | 3.4Mb  |
|  *lite::cv::face::detect::YOLO5Face*  |          yolov5face-l-640x640.onnx          |                 [YOLO5Face](https://github.com/deepcam-cn/yolov5-face)                  | 181Mb  |
|  *lite::cv::face::detect::YOLO5Face*  |          yolov5face-m-640x640.onnx          |                 [YOLO5Face](https://github.com/deepcam-cn/yolov5-face)                  |  83Mb  |
|  *lite::cv::face::detect::YOLO5Face*  |        yolov5face-n-0.5-320x320.onnx        |                 [YOLO5Face](https://github.com/deepcam-cn/yolov5-face)                  | 2.5Mb  |
|  *lite::cv::face::detect::YOLO5Face*  |        yolov5face-n-0.5-640x640.onnx        |                 [YOLO5Face](https://github.com/deepcam-cn/yolov5-face)                  | 4.6Mb  |
|  *lite::cv::face::detect::YOLO5Face*  |          yolov5face-n-640x640.onnx          |                 [YOLO5Face](https://github.com/deepcam-cn/yolov5-face)                  | 9.5Mb  |
|  *lite::cv::face::detect::YOLO5Face*  |          yolov5face-s-640x640.onnx          |                 [YOLO5Face](https://github.com/deepcam-cn/yolov5-face)                  |  30Mb  |
| *lite::cv::face::detect::FaceBoxesV2* |          faceboxesv2-640x640.onnx           |                [FaceBoxesV2](https://github.com/jhb86253817/FaceBoxesV2)                | 4.0Mb  |

#### Face Alignment.

<div id="lite.ai.toolkit.hub.onnx-face-alignment"></div>  


|                   Class                    |                     Pretrained ONNX Files                     |                  Rename or Converted From (Repo)                   |  Size   |
|:------------------------------------------:|:-------------------------------------------------------------:|:------------------------------------------------------------------:|:-------:|
|       *lite::cv::face::align::PFLD*        |                      pfld-106-lite.onnx                       | [pfld_106_...](https://github.com/Hsintao/pfld_106_face_landmarks) |  1.0Mb  |
|       *lite::cv::face::align::PFLD*        |                       pfld-106-v3.onnx                        | [pfld_106_...](https://github.com/Hsintao/pfld_106_face_landmarks) |  5.5Mb  |
|       *lite::cv::face::align::PFLD*        |                       pfld-106-v2.onnx                        | [pfld_106_...](https://github.com/Hsintao/pfld_106_face_landmarks) |  5.0Mb  |
|      *lite::cv::face::align::PFLD98*       |                    PFLD-pytorch-pfld.onnx                     |       [PFLD...](https://github.com/polarisZhao/PFLD-pytorch)       |  4.8Mb  |
|   *lite::cv::face::align::MobileNetV268*   |       pytorch_face_landmarks_landmark_detection_56.onnx       |  [...landmark](https://github.com/cunjian/pytorch_face_landmark)   |  9.4Mb  |
|  *lite::cv::face::align::MobileNetV2SE68*  | pytorch_face_landmarks_landmark_detection_56_se_external.onnx |  [...landmark](https://github.com/cunjian/pytorch_face_landmark)   |  11Mb   |
|      *lite::cv::face::align::PFLD68*       |               pytorch_face_landmarks_pfld.onnx                |  [...landmark](https://github.com/cunjian/pytorch_face_landmark)   |  2.8Mb  |
| *lite::cv::face::align::FaceLandmarks1000* |                     FaceLandmark1000.onnx                     |   [FaceLandm...](https://github.com/Single430/FaceLandmark1000)    |  2.0Mb  |
|     *lite::cv::face::align::PIPNet98*      |            pipnet_resnet18_10x19x32x256_aflw.onnx             |          [PIPNet](https://github.com/jhb86253817/PIPNet)           | 44.0Mb  |
|     *lite::cv::face::align::PIPNet68*      |            pipnet_resnet18_10x29x32x256_cofw.onnx             |          [PIPNet](https://github.com/jhb86253817/PIPNet)           | 44.0Mb  |
|     *lite::cv::face::align::PIPNet29*      |            pipnet_resnet18_10x68x32x256_300w.onnx             |          [PIPNet](https://github.com/jhb86253817/PIPNet)           | 44.0Mb  |
|     *lite::cv::face::align::PIPNet19*      |            pipnet_resnet18_10x98x32x256_wflw.onnx             |          [PIPNet](https://github.com/jhb86253817/PIPNet)           | 44.0Mb  |
|     *lite::cv::face::align::PIPNet98*      |            pipnet_resnet101_10x19x32x256_aflw.onnx            |          [PIPNet](https://github.com/jhb86253817/PIPNet)           | 150.0Mb |
|     *lite::cv::face::align::PIPNet68*      |            pipnet_resnet101_10x29x32x256_cofw.onnx            |          [PIPNet](https://github.com/jhb86253817/PIPNet)           | 150.0Mb |
|     *lite::cv::face::align::PIPNet29*      |            pipnet_resnet101_10x68x32x256_300w.onnx            |          [PIPNet](https://github.com/jhb86253817/PIPNet)           | 150.0Mb |
|     *lite::cv::face::align::PIPNet19*      |            pipnet_resnet101_10x98x32x256_wflw.onnx            |          [PIPNet](https://github.com/jhb86253817/PIPNet)           | 150.0Mb |



#### Face Attributes.

<div id="lite.ai.toolkit.hub.onnx-face-attributes"></div>  


|                   Class                   |                    Pretrained ONNX Files                     |                      Rename or Converted From (Repo)                      | Size  |
|:-----------------------------------------:|:------------------------------------------------------------:|:-------------------------------------------------------------------------:|:-----:|
|   *lite::cv::face::attr::AgeGoogleNet*    |                      age_googlenet.onnx                      |               [onnx-models](https://github.com/onnx/models)               | 23Mb  |
|  *lite::cv::face::attr::GenderGoogleNet*  |                    gender_googlenet.onnx                     |               [onnx-models](https://github.com/onnx/models)               | 23Mb  |
|  *lite::cv::face::attr::EmotionFerPlus*   |                    emotion-ferplus-7.onnx                    |               [onnx-models](https://github.com/onnx/models)               | 33Mb  |
|  *lite::cv::face::attr::EmotionFerPlus*   |                    emotion-ferplus-8.onnx                    |               [onnx-models](https://github.com/onnx/models)               | 33Mb  |
|     *lite::cv::face::attr::VGG16Age*      |               vgg_ilsvrc_16_age_imdb_wiki.onnx               |               [onnx-models](https://github.com/onnx/models)               | 514Mb |
|     *lite::cv::face::attr::VGG16Age*      |           vgg_ilsvrc_16_age_chalearn_iccv2015.onnx           |               [onnx-models](https://github.com/onnx/models)               | 514Mb |
|    *lite::cv::face::attr::VGG16Gender*    |             vgg_ilsvrc_16_gender_imdb_wiki.onnx              |               [onnx-models](https://github.com/onnx/models)               | 512Mb |
|      *lite::cv::face::attr::SSRNet*       |                         ssrnet.onnx                          |         [SSR_Net...](https://github.com/oukohou/SSR_Net_Pytorch)          | 190Kb |
| *lite::cv::face::attr::EfficientEmotion7* |           face-emotion-recognition-enet_b0_7.onnx            | [face-emo...](https://github.com/HSE-asavchenko/face-emotion-recognition) | 15Mb  |
| *lite::cv::face::attr::EfficientEmotion8* |      face-emotion-recognition-enet_b0_8_best_afew.onnx       | [face-emo...](https://github.com/HSE-asavchenko/face-emotion-recognition) | 15Mb  |
| *lite::cv::face::attr::EfficientEmotion8* |      face-emotion-recognition-enet_b0_8_best_vgaf.onnx       | [face-emo...](https://github.com/HSE-asavchenko/face-emotion-recognition) | 15Mb  |
|  *lite::cv::face::attr::MobileEmotion7*   |          face-emotion-recognition-mobilenet_7.onnx           | [face-emo...](https://github.com/HSE-asavchenko/face-emotion-recognition) | 13Mb  |
|  *lite::cv::face::attr::ReXNetEmotion7*   | face-emotion-recognition-affectnet_7_vggface2_rexnet150.onnx | [face-emo...](https://github.com/HSE-asavchenko/face-emotion-recognition) | 30Mb  |


#### Face Recognition.

<div id="lite.ai.toolkit.hub.onnx-face-recognition"></div>  


|                   Class                   |                  Pretrained ONNX Files                  |                    Rename or Converted From (Repo)                     | Size  |
|:-----------------------------------------:|:-------------------------------------------------------:|:----------------------------------------------------------------------:|:-----:|
|     *lite::cv::faceid::GlintArcFace*      |                ms1mv3_arcface_r100.onnx                 |       [insightface](https://github.com/deepinsight/insightface)        | 248Mb |
|     *lite::cv::faceid::GlintArcFace*      |                 ms1mv3_arcface_r50.onnx                 |       [insightface](https://github.com/deepinsight/insightface)        | 166Mb |
|     *lite::cv::faceid::GlintArcFace*      |                 ms1mv3_arcface_r34.onnx                 |       [insightface](https://github.com/deepinsight/insightface)        | 130Mb |
|     *lite::cv::faceid::GlintArcFace*      |                 ms1mv3_arcface_r18.onnx                 |       [insightface](https://github.com/deepinsight/insightface)        | 91Mb  |
|     *lite::cv::faceid::GlintCosFace*      |               glint360k_cosface_r100.onnx               |       [insightface](https://github.com/deepinsight/insightface)        | 248Mb |
|     *lite::cv::faceid::GlintCosFace*      |               glint360k_cosface_r50.onnx                |       [insightface](https://github.com/deepinsight/insightface)        | 166Mb |
|     *lite::cv::faceid::GlintCosFace*      |               glint360k_cosface_r34.onnx                |       [insightface](https://github.com/deepinsight/insightface)        | 130Mb |
|     *lite::cv::faceid::GlintCosFace*      |               glint360k_cosface_r18.onnx                |       [insightface](https://github.com/deepinsight/insightface)        | 91Mb  |
|    *lite::cv::faceid::GlintPartialFC*     |             partial_fc_glint360k_r100.onnx              |       [insightface](https://github.com/deepinsight/insightface)        | 248Mb |
|    *lite::cv::faceid::GlintPartialFC*     |              partial_fc_glint360k_r50.onnx              |       [insightface](https://github.com/deepinsight/insightface)        | 91Mb  |
|        *lite::cv::faceid::FaceNet*        |              facenet_vggface2_resnet.onnx               |       [facenet...](https://github.com/timesler/facenet-pytorch)        | 89Mb  |
|        *lite::cv::faceid::FaceNet*        |            facenet_casia-webface_resnet.onnx            |       [facenet...](https://github.com/timesler/facenet-pytorch)        | 89Mb  |
|     *lite::cv::faceid::FocalArcFace*      |              focal-arcface-ms1m-ir152.onnx              |   [face.evoLVe...](https://github.com/ZhaoJ9014/face.evoLVe.PyTorch)   | 269Mb |
|     *lite::cv::faceid::FocalArcFace*      |          focal-arcface-ms1m-ir50-epoch120.onnx          |   [face.evoLVe...](https://github.com/ZhaoJ9014/face.evoLVe.PyTorch)   | 166Mb |
|     *lite::cv::faceid::FocalArcFace*      |          focal-arcface-ms1m-ir50-epoch63.onnx           |   [face.evoLVe...](https://github.com/ZhaoJ9014/face.evoLVe.PyTorch)   | 166Mb |
|   *lite::cv::faceid::FocalAsiaArcFace*    |             focal-arcface-bh-ir50-asia.onnx             |   [face.evoLVe...](https://github.com/ZhaoJ9014/face.evoLVe.PyTorch)   | 166Mb |
| *lite::cv::faceid::TencentCurricularFace* |          Tencent_CurricularFace_Backbone.onnx           |               [TFace](https://github.com/Tencent/TFace)                | 249Mb |
|    *lite::cv::faceid::TencentCifpFace*    |        Tencent_Cifp_BUPT_Balancedface_IR_34.onnx        |               [TFace](https://github.com/Tencent/TFace)                | 130Mb |
|    *lite::cv::faceid::CenterLossFace*     |              CenterLossFace_epoch_100.onnx              |   [center-loss...](https://github.com/louis-she/center-loss.pytorch)   | 280Mb |
|      *lite::cv::faceid::SphereFace*       |                 sphere20a_20171020.onnx                 |      [sphere...](https://github.com/clcarwin/sphereface_pytorch)       | 86Mb  |
|    *lite::cv::faceid::PoseRobustFace*     |              dream_cfp_res50_end2end.onnx               |             [DREAM](https://github.com/penincillin/DREAM)              | 92Mb  |
|    *lite::cv::faceid::PoseRobustFace*     |              dream_ijba_res18_end2end.onnx              |             [DREAM](https://github.com/penincillin/DREAM)              | 43Mb  |
|  *lite::cv::faceid:NaivePoseRobustFace*   |               dream_cfp_res50_naive.onnx                |             [DREAM](https://github.com/penincillin/DREAM)              | 91Mb  |
|  *lite::cv::faceid:NaivePoseRobustFace*   |               dream_ijba_res18_naive.onnx               |             [DREAM](https://github.com/penincillin/DREAM)              | 43Mb  |
|     *lite::cv::faceid:MobileFaceNet*      |             MobileFaceNet_Pytorch_068.onnx              |   [MobileFace...](https://github.com/Xiaoccer/MobileFaceNet_Pytorch)   | 3.8Mb |
|    *lite::cv::faceid:CavaGhostArcFace*    |      cavaface_GhostNet_x1.3_Arcface_Epoch_24.onnx       |     [cavaface...](https://github.com/cavalleria/cavaface.pytorch)      | 15Mb  |
|    *lite::cv::faceid:CavaCombinedFace*    |        cavaface_IR_SE_100_Combined_Epoch_24.onnx        |     [cavaface...](https://github.com/cavalleria/cavaface.pytorch)      | 250Mb |
|   *lite::cv::faceid:MobileSEFocalFace*    | face_recognition.pytorch_Mobilenet_se_focal_121000.onnx | [face_recog...](https://github.com/grib0ed0v/face_recognition.pytorch) | 4.5Mb |


#### Head Pose Estimation.

<div id="lite.ai.toolkit.hub.onnx-head-pose-estimation"></div>  


|             Class              | Pretrained ONNX Files |                  Rename or Converted From (Repo)                   | Size  |
|:------------------------------:|:---------------------:|:------------------------------------------------------------------:|:-----:|
| *lite::cv::face::pose::FSANet* |    fsanet-var.onnx    | [...fsanet...](https://github.com/omasaht/headpose-fsanet-pytorch) | 1.2Mb |
| *lite::cv::face::pose::FSANet* |    fsanet-1x1.onnx    | [...fsanet...](https://github.com/omasaht/headpose-fsanet-pytorch) | 1.2Mb |


#### Segmentation.

<div id="lite.ai.toolkit.hub.onnx-segmentation"></div>  


|                    Class                     |     Pretrained ONNX Files     |         Rename or Converted From (Repo)          | Size  |
|:--------------------------------------------:|:-----------------------------:|:------------------------------------------------:|:-----:|
| *lite::cv::segmentation::DeepLabV3ResNet101* | deeplabv3_resnet101_coco.onnx | [torchvision](https://github.com/pytorch/vision) | 232Mb |
|    *lite::cv::segmentation::FCNResNet101*    |      fcn_resnet101.onnx       | [torchvision](https://github.com/pytorch/vision) | 207Mb |


#### Style Transfer.

<div id="lite.ai.toolkit.hub.onnx-style-transfer"></div>  

|                Class                 |   Pretrained ONNX Files    |        Rename or Converted From (Repo)        | Size  |
|:------------------------------------:|:--------------------------:|:---------------------------------------------:|:-----:|
| *lite::cv::style::FastStyleTransfer* |    style-mosaic-8.onnx     | [onnx-models](https://github.com/onnx/models) | 6.4Mb |
| *lite::cv::style::FastStyleTransfer* |     style-candy-9.onnx     | [onnx-models](https://github.com/onnx/models) | 6.4Mb |
| *lite::cv::style::FastStyleTransfer* |     style-udnie-8.onnx     | [onnx-models](https://github.com/onnx/models) | 6.4Mb |
| *lite::cv::style::FastStyleTransfer* |     style-udnie-9.onnx     | [onnx-models](https://github.com/onnx/models) | 6.4Mb |
| *lite::cv::style::FastStyleTransfer* |  style-pointilism-8.onnx   | [onnx-models](https://github.com/onnx/models) | 6.4Mb |
| *lite::cv::style::FastStyleTransfer* |  style-pointilism-9.onnx   | [onnx-models](https://github.com/onnx/models) | 6.4Mb |
| *lite::cv::style::FastStyleTransfer* | style-rain-princess-9.onnx | [onnx-models](https://github.com/onnx/models) | 6.4Mb |
| *lite::cv::style::FastStyleTransfer* | style-rain-princess-8.onnx | [onnx-models](https://github.com/onnx/models) | 6.4Mb |
| *lite::cv::style::FastStyleTransfer* |     style-candy-8.onnx     | [onnx-models](https://github.com/onnx/models) | 6.4Mb |
| *lite::cv::style::FastStyleTransfer* |    style-mosaic-9.onnx     | [onnx-models](https://github.com/onnx/models) | 6.4Mb |


#### Colorization.

<div id="lite.ai.toolkit.hub.onnx-colorization"></div>

|                Class                |   Pretrained ONNX Files   |              Rename or Converted From (Repo)              | Size  |
|:-----------------------------------:|:-------------------------:|:---------------------------------------------------------:|:-----:|
| *lite::cv::colorization::Colorizer* |   eccv16-colorizer.onnx   | [colorization](https://github.com/richzhang/colorization) | 123Mb |
| *lite::cv::colorization::Colorizer* | siggraph17-colorizer.onnx | [colorization](https://github.com/richzhang/colorization) | 129Mb |


#### Super Resolution.

<div id="lite.ai.toolkit.hub.onnx-super-resolution"></div>

|                Class                | Pretrained ONNX Files |              Rename or Converted From (Repo)              | Size  |
|:-----------------------------------:|:---------------------:|:---------------------------------------------------------:|:-----:|
| *lite::cv::resolution::SubPixelCNN* |   subpixel-cnn.onnx   | [...PIXEL...](https://github.com/niazwazir/SUB_PIXEL_CNN) | 234Kb |


#### Matting.

<div id="lite.ai.toolkit.hub.onnx-matting"></div>

|                    Class                    |                Pretrained ONNX Files                |                    Rename or Converted From (Repo)                     | Size  |
|:-------------------------------------------:|:---------------------------------------------------:|:----------------------------------------------------------------------:|:-----:|
|   *lite::cv::matting::RobustVideoMatting*   |              rvm_mobilenetv3_fp32.onnx              |  [RobustVideoMatting](https://github.com/PeterL1n/RobustVideoMatting)  | 14Mb  |
|   *lite::cv::matting::RobustVideoMatting*   |              rvm_mobilenetv3_fp16.onnx              |  [RobustVideoMatting](https://github.com/PeterL1n/RobustVideoMatting)  | 7.2Mb |
|   *lite::cv::matting::RobustVideoMatting*   |               rvm_resnet50_fp32.onnx                |  [RobustVideoMatting](https://github.com/PeterL1n/RobustVideoMatting)  | 50Mb  |
|   *lite::cv::matting::RobustVideoMatting*   |               rvm_resnet50_fp16.onnx                |  [RobustVideoMatting](https://github.com/PeterL1n/RobustVideoMatting)  | 100Mb |
|       *lite::cv::matting::MGMatting*        |               MGMatting-DIM-100k.onnx               |          [MGMatting](https://github.com/yucornetto/MGMatting)          | 113Mb |
|       *lite::cv::matting::MGMatting*        |               MGMatting-RWP-100k.onnx               |          [MGMatting](https://github.com/yucornetto/MGMatting)          | 113Mb |
|         *lite::cv::matting::MODNet*         | modnet_photographic_portrait_matting-1024x1024.onnx |               [MODNet](https://github.com/ZHKKKe/MODNet)               | 24Mb  |
|         *lite::cv::matting::MODNet*         | modnet_photographic_portrait_matting-1024x512.onnx  |               [MODNet](https://github.com/ZHKKKe/MODNet)               | 24Mb  |
|         *lite::cv::matting::MODNet*         |  modnet_photographic_portrait_matting-256x256.onnx  |               [MODNet](https://github.com/ZHKKKe/MODNet)               | 24Mb  |
|         *lite::cv::matting::MODNet*         |  modnet_photographic_portrait_matting-256x512.onnx  |               [MODNet](https://github.com/ZHKKKe/MODNet)               | 24Mb  |
|         *lite::cv::matting::MODNet*         | modnet_photographic_portrait_matting-512x1024.onnx  |               [MODNet](https://github.com/ZHKKKe/MODNet)               | 24Mb  |
|         *lite::cv::matting::MODNet*         |  modnet_photographic_portrait_matting-512x256.onnx  |               [MODNet](https://github.com/ZHKKKe/MODNet)               | 24Mb  |
|         *lite::cv::matting::MODNet*         |  modnet_photographic_portrait_matting-512x512.onnx  |               [MODNet](https://github.com/ZHKKKe/MODNet)               | 24Mb  |
|         *lite::cv::matting::MODNet*         |    modnet_webcam_portrait_matting-1024x1024.onnx    |               [MODNet](https://github.com/ZHKKKe/MODNet)               | 24Mb  |
|         *lite::cv::matting::MODNet*         |    modnet_webcam_portrait_matting-1024x512.onnx     |               [MODNet](https://github.com/ZHKKKe/MODNet)               | 24Mb  |
|         *lite::cv::matting::MODNet*         |     modnet_webcam_portrait_matting-256x256.onnx     |               [MODNet](https://github.com/ZHKKKe/MODNet)               | 24Mb  |
|         *lite::cv::matting::MODNet*         |     modnet_webcam_portrait_matting-256x512.onnx     |               [MODNet](https://github.com/ZHKKKe/MODNet)               | 24Mb  |
|         *lite::cv::matting::MODNet*         |    modnet_webcam_portrait_matting-512x1024.onnx     |               [MODNet](https://github.com/ZHKKKe/MODNet)               | 24Mb  |
|         *lite::cv::matting::MODNet*         |     modnet_webcam_portrait_matting-512x256.onnx     |               [MODNet](https://github.com/ZHKKKe/MODNet)               | 24Mb  |
|         *lite::cv::matting::MODNet*         |     modnet_webcam_portrait_matting-512x512.onnx     |               [MODNet](https://github.com/ZHKKKe/MODNet)               | 24Mb  |
|       *lite::cv::matting::MODNetDyn*        |      modnet_photographic_portrait_matting.onnx      |               [MODNet](https://github.com/ZHKKKe/MODNet)               | 24Mb  |
|       *lite::cv::matting::MODNetDyn*        |         modnet_webcam_portrait_matting.onnx         |               [MODNet](https://github.com/ZHKKKe/MODNet)               | 24Mb  |
|  *lite::cv::matting::BackgroundMattingV2*   |         BGMv2_mobilenetv2-256x256-full.onnx         | [BackgroundMattingV2](https://github.com/PeterL1n/BackgroundMattingV2) | 20Mb  |
|  *lite::cv::matting::BackgroundMattingV2*   |         BGMv2_mobilenetv2-512x512-full.onnx         | [BackgroundMattingV2](https://github.com/PeterL1n/BackgroundMattingV2) | 20Mb  |
|  *lite::cv::matting::BackgroundMattingV2*   |        BGMv2_mobilenetv2-1080x1920-full.onnx        | [BackgroundMattingV2](https://github.com/PeterL1n/BackgroundMattingV2) | 20Mb  |
|  *lite::cv::matting::BackgroundMattingV2*   |        BGMv2_mobilenetv2-2160x3840-full.onnx        | [BackgroundMattingV2](https://github.com/PeterL1n/BackgroundMattingV2) | 20Mb  |
|  *lite::cv::matting::BackgroundMattingV2*   |         BGMv2_resnet50-1080x1920-full.onnx          | [BackgroundMattingV2](https://github.com/PeterL1n/BackgroundMattingV2) | 20Mb  |
|  *lite::cv::matting::BackgroundMattingV2*   |         BGMv2_resnet50-2160x3840-full.onnx          | [BackgroundMattingV2](https://github.com/PeterL1n/BackgroundMattingV2) | 20Mb  |
|  *lite::cv::matting::BackgroundMattingV2*   |         BGMv2_resnet101-2160x3840-full.onnx         | [BackgroundMattingV2](https://github.com/PeterL1n/BackgroundMattingV2) | 154Mb |
| *lite::cv::matting::BackgroundMattingV2Dyn* |          BGMv2_mobilenetv2_4k_dynamic.onnx          | [BackgroundMattingV2](https://github.com/PeterL1n/BackgroundMattingV2) | 157Mb |
| *lite::cv::matting::BackgroundMattingV2Dyn* |          BGMv2_mobilenetv2_hd_dynamic.onnx          | [BackgroundMattingV2](https://github.com/PeterL1n/BackgroundMattingV2) | 230Mb |

</details>

## 6. 应用案例

<div id="lite.ai.toolkit-Examples-for-Lite.AI.ToolKit"></div>

更多的应用案例详见[examples](https://github.com/DefTruth/lite.ai.toolkit/tree/main/examples/lite/cv) 。点击 ▶️ 可以看到该主题下更多的案例。

<div id="lite.ai.toolkit-object-detection"></div>

#### 案例0: 使用[YOLOv5](https://github.com/ultralytics/yolov5) 进行目标检测。请从Model-Zoo[<sup>2</sup>](#lite.ai.toolkit-2) 下载模型文件。
```c++
#include "lite/lite.h"

static void test_default()
{
  std::string onnx_path = "../../../hub/onnx/cv/yolov5s.onnx";
  std::string test_img_path = "../../../examples/lite/resources/test_lite_yolov5_1.jpg";
  std::string save_img_path = "../../../logs/test_lite_yolov5_1.jpg";

  auto *yolov5 = new lite::cv::detection::YoloV5(onnx_path); 
  std::vector<lite::types::Boxf> detected_boxes;
  cv::Mat img_bgr = cv::imread(test_img_path);
  yolov5->detect(img_bgr, detected_boxes);
  
  lite::utils::draw_boxes_inplace(img_bgr, detected_boxes);
  cv::imwrite(save_img_path, img_bgr);  
  
  delete yolov5;
}
```

输出的结果是:
<div align='center'>
  <img src='logs/test_lite_yolov5_1.jpg' height="256px">
  <img src='logs/test_lite_yolov5_2.jpg' height="256px">
</div>

或者你可以使用最新的 🔥🔥 ! YOLO 系列检测器[YOLOX](https://github.com/Megvii-BaseDetection/YOLOX) 或 [YoloR](https://github.com/WongKinYiu/yolor) ，它们会获得接近的结果。

更多可用的通用目标检测器（80类、COCO）:  
```c++
auto *detector = new lite::cv::detection::YoloX(onnx_path);  // Newest YOLO detector !!! 2021-07
auto *detector = new lite::cv::detection::YoloV4(onnx_path); 
auto *detector = new lite::cv::detection::YoloV3(onnx_path); 
auto *detector = new lite::cv::detection::TinyYoloV3(onnx_path); 
auto *detector = new lite::cv::detection::SSD(onnx_path); 
auto *detector = new lite::cv::detection::YoloV5(onnx_path); 
auto *detector = new lite::cv::detection::YoloR(onnx_path);  // Newest YOLO detector !!! 2021-05
auto *detector = new lite::cv::detection::TinyYoloV4VOC(onnx_path); 
auto *detector = new lite::cv::detection::TinyYoloV4COCO(onnx_path); 
auto *detector = new lite::cv::detection::ScaledYoloV4(onnx_path); 
auto *detector = new lite::cv::detection::EfficientDet(onnx_path); 
auto *detector = new lite::cv::detection::EfficientDetD7(onnx_path); 
auto *detector = new lite::cv::detection::EfficientDetD8(onnx_path); 
auto *detector = new lite::cv::detection::YOLOP(onnx_path);
auto *detector = new lite::cv::detection::NanoDet(onnx_path); // Super fast and tiny!
auto *detector = new lite::cv::detection::NanoDetPlus(onnx_path); // Super fast and tiny! 2021/12/25
auto *detector = new lite::cv::detection::NanoDetEfficientNetLite(onnx_path); // Super fast and tiny!
```

****

<div id="lite.ai.toolkit-matting"></div>  

#### 案例1: 使用[RobustVideoMatting2021🔥🔥🔥](https://github.com/PeterL1n/RobustVideoMatting) 进行视频抠图。请从Model-Zoo[<sup>2</sup>](#lite.ai.toolkit-2) 下载模型文件。

```c++
#include "lite/lite.h"

static void test_default()
{
  std::string onnx_path = "../../../hub/onnx/cv/rvm_mobilenetv3_fp32.onnx";
  std::string video_path = "../../../examples/lite/resources/test_lite_rvm_0.mp4";
  std::string output_path = "../../../logs/test_lite_rvm_0.mp4";
  
  auto *rvm = new lite::cv::matting::RobustVideoMatting(onnx_path, 16); // 16 threads
  std::vector<lite::types::MattingContent> contents;
  
  // 1. video matting.
  rvm->detect_video(video_path, output_path, contents, false, 0.4f);
  
  delete rvm;
}
```
输出的结果是:

<div align='center'>
  <img src='docs/resources/interviewi.gif' height="80px" width="150px">
  <img src='docs/resources/interview.gif' height="80px" width="150px">  
  <img src='docs/resources/dance3i.gif' height="80px" width="150px">
  <img src='docs/resources/dance3.gif' height="80px" width="150px">
  <br>
  <img src='docs/resources/teslai.gif' height="80px" width="150px">
  <img src='docs/resources/tesla.gif' height="80px" width="150px">  
  <img src='docs/resources/b5i.gif' height="80px" width="150px">
  <img src='docs/resources/b5.gif' height="80px" width="150px">
</div>

更多可用的抠图模型（图片抠图、视频抠图、trimap/mask-free、trimap/mask-based）:
```c++
auto *matting = new lite::cv::matting::RobustVideoMatting:(onnx_path);  //  WACV 2022.
auto *matting = new lite::cv::matting::MGMatting(onnx_path); // CVPR 2021
auto *matting = new lite::cv::matting::MODNet(onnx_path); // AAAI 2022
auto *matting = new lite::cv::matting::MODNetDyn(onnx_path); // AAAI 2022 Dynamic Shape Inference.
auto *matting = new lite::cv::matting::BackgroundMattingV2(onnx_path); // CVPR 2020 
auto *matting = new lite::cv::matting::BackgroundMattingV2Dyn(onnx_path); // CVPR 2020 Dynamic Shape Inference.
```

****

<div id="lite.ai.toolkit-face-alignment"></div>

#### 案例2: 使用[FaceLandmarks1000](https://github.com/Single430/FaceLandmark1000) 进行人脸1000关键点检测。请从Model-Zoo[<sup>2</sup>](#lite.ai.toolkit-2) 下载模型文件。
```c++
#include "lite/lite.h"

static void test_default()
{
  std::string onnx_path = "../../../hub/onnx/cv/FaceLandmark1000.onnx";
  std::string test_img_path = "../../../examples/lite/resources/test_lite_face_landmarks_0.png";
  std::string save_img_path = "../../../logs/test_lite_face_landmarks_1000.jpg";
    
  auto *face_landmarks_1000 = new lite::cv::face::align::FaceLandmark1000(onnx_path);

  lite::types::Landmarks landmarks;
  cv::Mat img_bgr = cv::imread(test_img_path);
  face_landmarks_1000->detect(img_bgr, landmarks);
  lite::utils::draw_landmarks_inplace(img_bgr, landmarks);
  cv::imwrite(save_img_path, img_bgr);
  
  delete face_landmarks_1000;
}
```
输出的结果是:
<div align='center'>
  <img src='logs/test_lite_face_landmarks_1000.jpg' height="224px" width="224px">
  <img src='logs/test_lite_face_landmarks_1000_2.jpg' height="224px" width="224px">
  <img src='logs/test_lite_face_landmarks_1000_0.jpg' height="224px" width="224px">
</div>    

更多可用的人脸关键点检测器（68点、98点、106点、1000点）:
```c++
auto *align = new lite::cv::face::align::PFLD(onnx_path);  // 106 landmarks, 1.0Mb only!
auto *align = new lite::cv::face::align::PFLD98(onnx_path);  // 98 landmarks, 4.8Mb only!
auto *align = new lite::cv::face::align::PFLD68(onnx_path);  // 68 landmarks, 2.8Mb only!
auto *align = new lite::cv::face::align::MobileNetV268(onnx_path);  // 68 landmarks, 9.4Mb only!
auto *align = new lite::cv::face::align::MobileNetV2SE68(onnx_path);  // 68 landmarks, 11Mb only!
auto *align = new lite::cv::face::align::FaceLandmark1000(onnx_path);  // 1000 landmarks, 2.0Mb only!
auto *align = new lite::cv::face::align::PIPNet98(onnx_path);  // 98 landmarks, CVPR2021!
auto *align = new lite::cv::face::align::PIPNet68(onnx_path);  // 68 landmarks, CVPR2021!
auto *align = new lite::cv::face::align::PIPNet29(onnx_path);  // 29 landmarks, CVPR2021!
auto *align = new lite::cv::face::align::PIPNet19(onnx_path);  // 19 landmarks, CVPR2021!
```

****  

<div id="lite.ai.toolkit-colorization"></div>

#### 案例3: 使用[colorization](https://github.com/richzhang/colorization) 进行图像着色。请从Model-Zoo[<sup>2</sup>](#lite.ai.toolkit-2) 下载模型文件。
```c++
#include "lite/lite.h"

static void test_default()
{
  std::string onnx_path = "../../../hub/onnx/cv/eccv16-colorizer.onnx";
  std::string test_img_path = "../../../examples/lite/resources/test_lite_colorizer_1.jpg";
  std::string save_img_path = "../../../logs/test_lite_eccv16_colorizer_1.jpg";
  
  auto *colorizer = new lite::cv::colorization::Colorizer(onnx_path);
  
  cv::Mat img_bgr = cv::imread(test_img_path);
  lite::types::ColorizeContent colorize_content;
  colorizer->detect(img_bgr, colorize_content);
  
  if (colorize_content.flag) cv::imwrite(save_img_path, colorize_content.mat);
  delete colorizer;
}
```
输出的结果是:

<div align='center'>
  <img src='examples/lite/resources/test_lite_colorizer_1.jpg' height="224px" width="224px">
  <img src='examples/lite/resources/test_lite_colorizer_2.jpg' height="224px" width="224px">
  <img src='examples/lite/resources/test_lite_colorizer_3.jpg' height="224px" width="224px">  
  <br> 
  <img src='logs/test_lite_siggraph17_colorizer_1.jpg' height="224px" width="224px">
  <img src='logs/test_lite_siggraph17_colorizer_2.jpg' height="224px" width="224px">
  <img src='logs/test_lite_siggraph17_colorizer_3.jpg' height="224px" width="224px">
</div>    

更多可用的着色器模型（灰度图转彩色图）:
```c++
auto *colorizer = new lite::cv::colorization::Colorizer(onnx_path);
```

****

<div id="lite.ai.toolkit-face-recognition"></div>  

#### 案例4: 使用[ArcFace](https://github.com/deepinsight/insightface/tree/master/recognition/arcface_torch) 进行人脸识别。请从Model-Zoo[<sup>2</sup>](#lite.ai.toolkit-2) 下载模型文件。

```c++
#include "lite/lite.h"

static void test_default()
{
  std::string onnx_path = "../../../hub/onnx/cv/ms1mv3_arcface_r100.onnx";
  std::string test_img_path0 = "../../../examples/lite/resources/test_lite_faceid_0.png";
  std::string test_img_path1 = "../../../examples/lite/resources/test_lite_faceid_1.png";
  std::string test_img_path2 = "../../../examples/lite/resources/test_lite_faceid_2.png";

  auto *glint_arcface = new lite::cv::faceid::GlintArcFace(onnx_path);

  lite::types::FaceContent face_content0, face_content1, face_content2;
  cv::Mat img_bgr0 = cv::imread(test_img_path0);
  cv::Mat img_bgr1 = cv::imread(test_img_path1);
  cv::Mat img_bgr2 = cv::imread(test_img_path2);
  glint_arcface->detect(img_bgr0, face_content0);
  glint_arcface->detect(img_bgr1, face_content1);
  glint_arcface->detect(img_bgr2, face_content2);

  if (face_content0.flag && face_content1.flag && face_content2.flag)
  {
    float sim01 = lite::utils::math::cosine_similarity<float>(
        face_content0.embedding, face_content1.embedding);
    float sim02 = lite::utils::math::cosine_similarity<float>(
        face_content0.embedding, face_content2.embedding);
    std::cout << "Detected Sim01: " << sim  << " Sim02: " << sim02 << std::endl;
  }

  delete glint_arcface;
}
```

输出的结果是:
<div align='center'>
  <img src='examples/lite/resources/test_lite_arcface_resnet_0.png' height="224px" width="224px">
  <img src='examples/lite/resources/test_lite_arcface_resnet_1.png' height="224px" width="224px">
  <img src='examples/lite/resources/test_lite_arcface_resnet_2.png' height="224px" width="224px">
</div>  

> Detected Sim01: 0.721159  Sim02: -0.0626267

更多可用的人脸识别模型（人脸特征提取）:
```c++
auto *recognition = new lite::cv::faceid::GlintCosFace(onnx_path);  // DeepGlint(insightface)
auto *recognition = new lite::cv::faceid::GlintArcFace(onnx_path);  // DeepGlint(insightface)
auto *recognition = new lite::cv::faceid::GlintPartialFC(onnx_path); // DeepGlint(insightface)
auto *recognition = new lite::cv::faceid::FaceNet(onnx_path);
auto *recognition = new lite::cv::faceid::FocalArcFace(onnx_path);
auto *recognition = new lite::cv::faceid::FocalAsiaArcFace(onnx_path);
auto *recognition = new lite::cv::faceid::TencentCurricularFace(onnx_path); // Tencent(TFace)
auto *recognition = new lite::cv::faceid::TencentCifpFace(onnx_path); // Tencent(TFace)
auto *recognition = new lite::cv::faceid::CenterLossFace(onnx_path);
auto *recognition = new lite::cv::faceid::SphereFace(onnx_path);
auto *recognition = new lite::cv::faceid::PoseRobustFace(onnx_path);
auto *recognition = new lite::cv::faceid::NaivePoseRobustFace(onnx_path);
auto *recognition = new lite::cv::faceid::MobileFaceNet(onnx_path); // 3.8Mb only !
auto *recognition = new lite::cv::faceid::CavaGhostArcFace(onnx_path);
auto *recognition = new lite::cv::faceid::CavaCombinedFace(onnx_path);
auto *recognition = new lite::cv::faceid::MobileSEFocalFace(onnx_path); // 4.5Mb only !
```

****

<div id="lite.ai.toolkit-face-detection"></div>

#### 案例5: 使用[SCRFD 2021](https://github.com/deepinsight/insightface/blob/master/detection/scrfd/) 进行人脸检测。请从Model-Zoo[<sup>2</sup>](#lite.ai.toolkit-2) 下载模型文件。
```c++
#include "lite/lite.h"

static void test_default()
{
  std::string onnx_path = "../../../hub/onnx/cv/scrfd_2.5g_bnkps_shape640x640.onnx";
  std::string test_img_path = "../../../examples/lite/resources/test_lite_face_detector.jpg";
  std::string save_img_path = "../../../logs/test_lite_scrfd.jpg";
  
  auto *scrfd = new lite::cv::face::detect::SCRFD(onnx_path);
  
  std::vector<lite::types::BoxfWithLandmarks> detected_boxes;
  cv::Mat img_bgr = cv::imread(test_img_path);
  scrfd->detect(img_bgr, detected_boxes);
  
  lite::utils::draw_boxes_with_landmarks_inplace(img_bgr, detected_boxes);
  cv::imwrite(save_img_path, img_bgr);
  
  std::cout << "Default Version Done! Detected Face Num: " << detected_boxes.size() << std::endl;
  
  delete scrfd;
}
```
输出的结果是:
<div align='center'>
  <img src='docs/resources/scrfd.jpg' height="224px" width="224px">
  <img src='docs/resources/scrfd_2.jpg' height="224px" width="224px">
  <img src='docs/resources/scrfd_3.jpg' height="224px" width="224px">
</div>  

更多可用的人脸检测器（轻量级人脸检测器）:
```c++
auto *detector = new lite::face::detect::UltraFace(onnx_path);  // 1.1Mb only !
auto *detector = new lite::face::detect::FaceBoxes(onnx_path);  // 3.8Mb only ! 
auto *detector = new lite::face::detect::FaceBoxesv2(onnx_path);  // 4.0Mb only ! 
auto *detector = new lite::face::detect::RetinaFace(onnx_path);  // 1.6Mb only ! CVPR2020
auto *detector = new lite::face::detect::SCRFD(onnx_path);  // 2.5Mb only ! CVPR2021, Super fast and accurate!!
auto *detector = new lite::face::detect::YOLO5Face(onnx_path);  // 2021, Super fast and accurate!!
```

<div id="lite.ai.toolkit-segmentation"></div>  

#### 案例6: 使用 [DeepLabV3ResNet101](https://pytorch.org/hub/pytorch_vision_deeplabv3_resnet101/) 进行语义分割. 请从Model-Zoo[<sup>2</sup>](#lite.ai.toolkit-2) 下载模型文件。
```c++
#include "lite/lite.h"

static void test_default()
{
  std::string onnx_path = "../../../hub/onnx/cv/deeplabv3_resnet101_coco.onnx";
  std::string test_img_path = "../../../examples/lite/resources/test_lite_deeplabv3_resnet101.png";
  std::string save_img_path = "../../../logs/test_lite_deeplabv3_resnet101.jpg";

  auto *deeplabv3_resnet101 = new lite::cv::segmentation::DeepLabV3ResNet101(onnx_path, 16); // 16 threads

  lite::types::SegmentContent content;
  cv::Mat img_bgr = cv::imread(test_img_path);
  deeplabv3_resnet101->detect(img_bgr, content);

  if (content.flag)
  {
    cv::Mat out_img;
    cv::addWeighted(img_bgr, 0.2, content.color_mat, 0.8, 0., out_img);
    cv::imwrite(save_img_path, out_img);
    if (!content.names_map.empty())
    {
      for (auto it = content.names_map.begin(); it != content.names_map.end(); ++it)
      {
        std::cout << it->first << " Name: " << it->second << std::endl;
      }
    }
  }
  delete deeplabv3_resnet101;
}
```

输出的结果是:  
<div align='center'>
  <img src='examples/lite/resources/test_lite_deeplabv3_resnet101.png' height="256px">
  <img src='logs/test_lite_deeplabv3_resnet101.jpg' height="256px">
</div> 

更多可用的语义分割模型（人像分割、实例分割）:
```c++
auto *segment = new lite::cv::segmentation::FCNResNet101(onnx_path);
auto *segment = new lite::cv::segmentation::DeepLabV3ResNet101(onnx_path);
```

<div id="lite.ai.toolkit-face-attributes-analysis"></div>    

#### 案例7: 使用 [SSRNet](https://github.com/oukohou/SSR_Net_Pytorch) 进行年龄估计. 请从Model-Zoo[<sup>2</sup>](#lite.ai.toolkit-2) 下载模型文件。
```c++
#include "lite/lite.h"

static void test_default()
{
  std::string onnx_path = "../../../hub/onnx/cv/ssrnet.onnx";
  std::string test_img_path = "../../../examples/lite/resources/test_lite_ssrnet.jpg";
  std::string save_img_path = "../../../logs/test_lite_ssrnet.jpg";

  lite::cv::face::attr::SSRNet *ssrnet = new lite::cv::face::attr::SSRNet(onnx_path);

  lite::types::Age age;
  cv::Mat img_bgr = cv::imread(test_img_path);
  ssrnet->detect(img_bgr, age);
  lite::utils::draw_age_inplace(img_bgr, age);
  cv::imwrite(save_img_path, img_bgr);
  std::cout << "Default Version Done! Detected SSRNet Age: " << age.age << std::endl;

  delete ssrnet;
}
```
输出的结果是:  
<div align='center'>
  <img src='logs/test_lite_ssrnet.jpg' height="224px" width="224px">
  <img src='logs/test_lite_gender_googlenet.jpg' height="224px" width="224px">
  <img src='logs/test_lite_emotion_ferplus.jpg' height="224px" width="224px">
</div>    

更多可用的人脸属性识别模型（性别、年龄、情绪）:
```c++
auto *attribute = new lite::cv::face::attr::AgeGoogleNet(onnx_path);  
auto *attribute = new lite::cv::face::attr::GenderGoogleNet(onnx_path); 
auto *attribute = new lite::cv::face::attr::EmotionFerPlus(onnx_path);
auto *attribute = new lite::cv::face::attr::VGG16Age(onnx_path);
auto *attribute = new lite::cv::face::attr::VGG16Gender(onnx_path);
auto *attribute = new lite::cv::face::attr::EfficientEmotion7(onnx_path); // 7 emotions, 15Mb only!
auto *attribute = new lite::cv::face::attr::EfficientEmotion8(onnx_path); // 8 emotions, 15Mb only!
auto *attribute = new lite::cv::face::attr::MobileEmotion7(onnx_path); // 7 emotions, 13Mb only!
auto *attribute = new lite::cv::face::attr::ReXNetEmotion7(onnx_path); // 7 emotions
auto *attribute = new lite::cv::face::attr::SSRNet(onnx_path); // age estimation, 190kb only!!!
```


<div id="lite.ai.toolkit-image-classification"></div>   

#### 案例8: 使用 [DenseNet](https://pytorch.org/hub/pytorch_vision_densenet/) 进行图片1000分类. 请从Model-Zoo[<sup>2</sup>](#lite.ai.toolkit-2) 下载模型文件。
```c++
#include "lite/lite.h"

static void test_default()
{
  std::string onnx_path = "../../../hub/onnx/cv/densenet121.onnx";
  std::string test_img_path = "../../../examples/lite/resources/test_lite_densenet.jpg";

  auto *densenet = new lite::cv::classification::DenseNet(onnx_path);

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
  }
  delete densenet;
}
```

输出的结果是:  
<div align='center'>
  <img src='examples/lite/resources/test_lite_densenet.jpg' height="224px" width="224px">
  <img src='logs/test_lite_densenet.png' height="224px" width="500px">
</div>  

更多可用的图像分类模型（1000类）:
```c++
auto *classifier = new lite::cv::classification::EfficientNetLite4(onnx_path);  
auto *classifier = new lite::cv::classification::ShuffleNetV2(onnx_path); // 8.7Mb only!
auto *classifier = new lite::cv::classification::GhostNet(onnx_path);
auto *classifier = new lite::cv::classification::HdrDNet(onnx_path);
auto *classifier = new lite::cv::classification::IBNNet(onnx_path);
auto *classifier = new lite::cv::classification::MobileNetV2(onnx_path); // 13Mb only!
auto *classifier = new lite::cv::classification::ResNet(onnx_path); 
auto *classifier = new lite::cv::classification::ResNeXt(onnx_path);
```


<div id="lite.ai.toolkit-head-pose-estimation"></div>   

#### 案例9: 使用 [FSANet](https://github.com/omasaht/headpose-fsanet-pytorch) 进行头部姿态识别. 请从Model-Zoo[<sup>2</sup>](#lite.ai.toolkit-2) 下载模型文件。

```c++
#include "lite/lite.h"

static void test_default()
{
  std::string onnx_path = "../../../hub/onnx/cv/fsanet-var.onnx";
  std::string test_img_path = "../../../examples/lite/resources/test_lite_fsanet.jpg";
  std::string save_img_path = "../../../logs/test_lite_fsanet.jpg";

  auto *fsanet = new lite::cv::face::pose::FSANet(onnx_path);
  cv::Mat img_bgr = cv::imread(test_img_path);
  lite::types::EulerAngles euler_angles;
  fsanet->detect(img_bgr, euler_angles);
  
  if (euler_angles.flag)
  {
    lite::utils::draw_axis_inplace(img_bgr, euler_angles);
    cv::imwrite(save_img_path, img_bgr);
    std::cout << "yaw:" << euler_angles.yaw << " pitch:" << euler_angles.pitch << " row:" << euler_angles.roll << std::endl;
  }
  delete fsanet;
}
```

输出的结果是:  
<div align='center'>
  <img src='logs/test_lite_fsanet.jpg' height="224px" width="224px">
  <img src='logs/test_lite_fsanet_2.jpg' height="224px" width="224px">
  <img src='logs/test_lite_fsanet_3.jpg' height="224px" width="224px">
</div>  

更多可用的头部姿态识别模型（欧拉角、yaw、pitch、roll）:
```c++
auto *pose = new lite::cv::face::pose::FSANet(onnx_path); // 1.2Mb only!
```

<div id="lite.ai.toolkit-style-transfer"></div>  

#### 案例10: 使用 [FastStyleTransfer](https://github.com/onnx/models/tree/master/vision/style_transfer/fast_neural_style) 进行风格迁移. 请从Model-Zoo[<sup>2</sup>](#lite.ai.toolkit-2) 下载模型文件。
```c++
#include "lite/lite.h"

static void test_default()
{
  std::string onnx_path = "../../../hub/onnx/cv/style-candy-8.onnx";
  std::string test_img_path = "../../../examples/lite/resources/test_lite_fast_style_transfer.jpg";
  std::string save_img_path = "../../../logs/test_lite_fast_style_transfer_candy.jpg";
  
  auto *fast_style_transfer = new lite::cv::style::FastStyleTransfer(onnx_path);
 
  lite::types::StyleContent style_content;
  cv::Mat img_bgr = cv::imread(test_img_path);
  fast_style_transfer->detect(img_bgr, style_content);

  if (style_content.flag) cv::imwrite(save_img_path, style_content.mat);
  delete fast_style_transfer;
}
```
输出的结果是:  

<div align='center'>
  <img src='examples/lite/resources/test_lite_fast_style_transfer.jpg' height="224px">
  <img src='logs/test_lite_fast_style_transfer_candy.jpg' height="224px">
  <img src='logs/test_lite_fast_style_transfer_mosaic.jpg' height="224px">  
  <br> 
  <img src='logs/test_lite_fast_style_transfer_pointilism.jpg' height="224px">
  <img src='logs/test_lite_fast_style_transfer_rain_princes.jpg' height="224px">
  <img src='logs/test_lite_fast_style_transfer_udnie.jpg' height="224px">
</div>

更多可用的风格迁移模型（自然风格迁移、其他）:
```c++
auto *transfer = new lite::cv::style::FastStyleTransfer(onnx_path); // 6.4Mb only
```

## 7. 开源协议

<div id="lite.ai.toolkit-License"></div>

[Lite.Ai.ToolKit](#lite.ai.toolkit-Introduction) 的代码采用GPL-3.0协议。


## 8. 引用参考

<div id="lite.ai.toolkit-References"></div>

本项目参考了以下开源项目。

* [RobustVideoMatting](https://github.com/PeterL1n/RobustVideoMatting) (🔥🔥🔥new!!↑)
* [nanodet](https://github.com/RangiLyu/nanodet) (🔥🔥🔥↑)
* [YOLOX](https://github.com/Megvii-BaseDetection/YOLOX) (🔥🔥🔥new!!↑)
* [YOLOP](https://github.com/hustvl/YOLOP) (🔥🔥new!!↑)
* [YOLOR](https://github.com/WongKinYiu/yolor) (🔥🔥new!!↑)
* [ScaledYOLOv4](https://github.com/WongKinYiu/ScaledYOLOv4) (🔥🔥🔥↑)
* [insightface](https://github.com/deepinsight/insightface) (🔥🔥🔥↑)
* [yolov5](https://github.com/ultralytics/yolov5) (🔥🔥💥↑)
* [TFace](https://github.com/Tencent/TFace) (🔥🔥↑)
* [YOLOv4-pytorch](https://github.com/argusswift/YOLOv4-pytorch) (🔥🔥🔥↑)
* [Ultra-Light-Fast-Generic-Face-Detector-1MB](https://github.com/Linzaer/Ultra-Light-Fast-Generic-Face-Detector-1MB) (🔥🔥🔥↑)

<details>
<summary> 展开更多引用参考 </summary>  

* [headpose-fsanet-pytorch](https://github.com/omasaht/headpose-fsanet-pytorch) (🔥↑)
* [pfld_106_face_landmarks](https://github.com/Hsintao/pfld_106_face_landmarks) (🔥🔥↑)
* [onnx-models](https://github.com/onnx/models) (🔥🔥🔥↑)
* [SSR_Net_Pytorch](https://github.com/oukohou/SSR_Net_Pytorch) (🔥↑)
* [colorization](https://github.com/richzhang/colorization) (🔥🔥🔥↑)
* [SUB_PIXEL_CNN](https://github.com/niazwazir/SUB_PIXEL_CNN) (🔥↑)
* [torchvision](https://github.com/pytorch/vision) (🔥🔥🔥↑)
* [facenet-pytorch](https://github.com/timesler/facenet-pytorch) (🔥↑)
* [face.evoLVe.PyTorch](https://github.com/ZhaoJ9014/face.evoLVe.PyTorch) (🔥🔥🔥↑)
* [center-loss.pytorch](https://github.com/louis-she/center-loss.pytorch) (🔥🔥↑)
* [sphereface_pytorch](https://github.com/clcarwin/sphereface_pytorch) (🔥🔥↑)
* [DREAM](https://github.com/penincillin/DREAM) (🔥🔥↑)
* [MobileFaceNet_Pytorch](https://github.com/Xiaoccer/MobileFaceNet_Pytorch) (🔥🔥↑)
* [cavaface.pytorch](https://github.com/cavalleria/cavaface.pytorch) (🔥🔥↑)
* [CurricularFace](https://github.com/HuangYG123/CurricularFace) (🔥🔥↑)
* [face-emotion-recognition](https://github.com/HSE-asavchenko/face-emotion-recognition) (🔥↑)
* [face_recognition.pytorch](https://github.com/grib0ed0v/face_recognition.pytorch) (🔥🔥↑)
* [PFLD-pytorch](https://github.com/polarisZhao/PFLD-pytorch) (🔥🔥↑)
* [pytorch_face_landmark](https://github.com/cunjian/pytorch_face_landmark) (🔥🔥↑)
* [FaceLandmark1000](https://github.com/Single430/FaceLandmark1000) (🔥🔥↑)
* [Pytorch_Retinaface](https://github.com/biubug6/Pytorch_Retinaface) (🔥🔥🔥↑)
* [FaceBoxes](https://github.com/zisianw/FaceBoxes.PyTorch) (🔥🔥↑)

</details>    


## 9. 编译选项
未来会增加一些模型的[MNN](https://github.com/alibaba/MNN) 、[NCNN](https://github.com/Tencent/ncnn) 和 [TNN](https://github.com/Tencent/TNN) 支持，但由于算子兼容等原因，也无法确保所有被[ONNXRuntime C++](https://github.com/microsoft/onnxruntime) 支持的模型能够在[MNN](https://github.com/alibaba/MNN) 、[NCNN](https://github.com/Tencent/ncnn) 和 [TNN](https://github.com/Tencent/TNN) 下跑通。所以，如果您想使用本项目支持的所有模型，并且不在意*1~2ms*的性能差距的话，请使用ONNXRuntime版本的实现。[ONNXRuntime](https://github.com/microsoft/onnxruntime) 是本仓库默认的推理引擎。但是如果你确实希望编译支持[MNN](https://github.com/alibaba/MNN) 、[NCNN](https://github.com/Tencent/ncnn) 和 [TNN](https://github.com/Tencent/TNN) 支持的Lite.Ai.ToolKit动态库，你可以按照以下的步骤进行设置。

* 在`build.sh`中添加`DENABLE_MNN=ON` 、`DENABLE_NCNN=ON` 或 `DENABLE_TNN=ON`，比如
```shell
cd build && cmake \
  -DCMAKE_BUILD_TYPE=MinSizeRel \
  -DINCLUDE_OPENCV=ON \   # 是否打包OpenCV进lite.ai.toolkit，默认ON；否则，你需要单独设置OpenCV
  -DENABLE_MNN=ON \       # 是否编译MNN版本的模型， 默认OFF，目前只支持部分模型
  -DENABLE_NCNN=OFF \     # 是否编译NCNN版本的模型，默认OFF，目前只支持部分模型
  -DENABLE_TNN=OFF \      # 是否编译TNN版本的模型， 默认OFF，目前只支持部分模型
  .. && make -j8
```  
* 使用MNN、NCNN或TNN版本的接口，详见案例[demo](https://github.com/DefTruth/lite.ai.toolkit/blob/main/examples/lite/cv/test_lite_nanodet.cpp) ，比如
```C++
auto *nanodet = new lite::mnn::cv::detection::NanoDet(mnn_path);
auto *nanodet = new lite::tnn::cv::detection::NanoDet(proto_path, model_path);
auto *nanodet = new lite::ncnn::cv::detection::NanoDet(param_path, bin_path);
```

## 10. 如何添加您的模型
<div id="lite.ai.toolkit-Contribute"></div>  

如何添加您自己的模型以及成为贡献者？具体步骤请参考 [CONTRIBUTING.zh.md](https://github.com/DefTruth/lite.ai.toolkit/issues/191) ，或者，❤️不妨给个⭐️🌟star，这应该是最简单的支持方式了。

