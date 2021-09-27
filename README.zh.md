

## Lite.AI.ToolKit 🚀🚀🌟: 一个开箱即用的C++ AI模型工具箱

[![](https://img.shields.io/badge/MacOS-pass-brightgreen.svg)](https://github.com/DefTruth/lite.ai.toolkit/releases/tag/v0.1.0) ![](https://img.shields.io/badge/Linux-pass-brightgreen.svg) ![](https://img.shields.io/badge/Windows-pass-brightgreen.svg) [![](https://img.shields.io/badge/Version-0.1.0-green.svg)](https://github.com/DefTruth/lite.ai.toolkit/releases/tag/v0.1.0) ![](https://img.shields.io/badge/Language-C/C%2B%2B-orange.svg) ![](https://img.shields.io/badge/Device-GPU/CPU-yellow.svg) ![](https://img.shields.io/badge/License-GPLv3-blue.svg)


<div id="lite.ai.toolkit-Introduction"></div>  

<div align='center'>
  <img src='logs/test_lite_yolov5_1.jpg' height="100px" width="100px">
  <img src='docs/resources/efficientdet_d0.jpg' height="100px" width="100px">
  <img src='docs/resources/street.jpg' height="100px" width="100px">
  <img src='logs/test_lite_ultraface.jpg' height="100px" width="100px">
  <img src='logs/test_lite_face_landmarks_1000.jpg' height="100px" width="100px">
  <img src='logs/test_lite_fsanet.jpg' height="100px" width="100px">
  <img src='logs/test_lite_deeplabv3_resnet101.jpg' height="100px" width="100px">
  <img src='logs/test_lite_fast_style_transfer_mosaic.jpg' height="100px" width="100px"> 
  <br>
  <img src='docs/resources/teslai.gif' height="100px" width="100px">
  <img src='docs/resources/tesla.gif' height="100px" width="100px">
  <img src='docs/resources/dance3i.gif' height="100px" width="100px">
  <img src='docs/resources/dance3.gif' height="100px" width="100px">  
  <img src='docs/resources/yolop1.png' height="100px" width="100px">
  <img src='docs/resources/yolop1.gif' height="100px" width="100px">
  <img src='docs/resources/yolop2.png' height="100px" width="100px">
  <img src='docs/resources/yolop2.gif' height="100px" width="100px">

</div>    

![](https://img.shields.io/github/stars/DefTruth/lite.ai.toolkit.svg?style=social) ![](https://img.shields.io/github/forks/DefTruth/lite.ai.toolkit.svg?style=social) ![](https://img.shields.io/github/watchers/DefTruth/lite.ai.toolkit.svg?style=social)



---

*Lite.AI.ToolKit* 🚀🚀🌟: 一个轻量级的`C++` AI模型工具箱，用户友好，开箱即用。已经包括 *[70+](https://github.com/DefTruth/lite.ai.toolkit/tree/main/docs/hub/lite.ai.toolkit.hub.onnx.md)* 流行的开源模型，还会继续增加😎。这是一个根据个人兴趣整理的C++工具箱，目前主要集中在`检测、分割、抠图、识别和目标跟踪`等领域，包含了最新的RVM, YOLOX, YOLOP, YOLOR, YoloV5, DeepLabV3, ArcFace等模型。 *Lite.AI.ToolKit* 默认是基于 *[ONNXRuntime C++](https://github.com/microsoft/onnxruntime)* 推理引擎的，后期可能会加入对 *[ncnn](https://github.com/Tencent/ncnn)* 或 *[MNN](https://github.com/alibaba/MNN)* 的支持，目前主要考虑易用性。需要更高性能支持的小伙伴可以基于本项目提供的`C++`实现和`ONNX`文件进行优化~ 如果您有想添加到本项目的新模型，欢迎`PR` ~👏👋 

<p align="center"><a href="README.zh.md">English</a> | 中文 </p>

---  

### Lite.AI.ToolKit 🚀🚀🌟 的核心特性  

* ❤️ *用户友好，开箱即用。*  
  使用简单一致的调用语法，如*lite::cv::Type::Class*，详见[examples](#lite.ai.toolkit-Examples-for-Lite.AI.ToolKit).
  ```c++
    auto *yolox = new lite::cv::detection::YoloX("yolox_nano.onnx");  // 3.5Mb only !
    auto *yolov5 = new lite::cv::detection::YoloV5("yolov5s.onnx");  // for mobile device  
  ```
* ⚡ *少量依赖，构建容易。*  
  目前, 只依赖 *OpenCV* 和 *ONNXRuntime*，详见[build](#lite.ai.toolkit-Build-Lite.AI.ToolKit)。对于 MacOS，只需运行.👇
    ```shell
    git clone --depth=1 https://github.com/DefTruth/lite.ai.toolkit.git  # 最新源码
    cd lite.ai.toolkit && sh ./build.sh  # 对于MacOS, 你可以直接利用本项目包含的依赖库，无需重新编译
    ```
* ✅ *多平台编译支持，GPU/CPU支持。*  
  目前，支持 [MacOS/Linux/Windows](#lite.ai.toolkit-Introduction) 和 CPU/GPU。未来会对更多平台进行支持~

* ❤️ *众多的算法模块，且持续更新*  
  目前，包括 10+ 算法模块和 *[70+](https://github.com/DefTruth/lite.ai.toolkit/tree/main/docs/hub/lite.ai.toolkit.hub.onnx.md)* 流行的开源模型, 涵盖[目标检测](#lite.ai.toolkit-object-detection)、[人脸检测](#lite.ai.toolkit-face-detection)、[人脸识别](#lite.ai.toolkit-face-recognition)、[语义分割](#lite.ai.toolkit-segmentation)、[抠图](#lite.ai.toolkit-matting)等领域。详见 [Model Zoo](#lite.ai.toolkit-Model-Zoo)。更多的新模型将会不断地加入进来 ~ 😎

* ✅ *最新的发行版本，以及详细的文档。*

  |发行版本|快速开始|详细用法| 
    |:---:|:---:|:---:| 
  | 👉 [lite.ai.toolkit.macos.v0.1.0](https://github.com/DefTruth/lite.ai.toolkit.demo/tree/main/releases/macos/v0.1.0) |  👉 [lite.ai.toolkit.demo](https://github.com/DefTruth/lite.ai.toolkit.demo) & [Quick Start Examples](#lite.ai.toolkit-Examples-for-Lite.AI.ToolKit) |  👉 [lite.ai.toolkit.examples](https://github.com/DefTruth/lite.ai.toolkit/tree/main/examples/lite/cv) |

---

## 重要更新 !!!

* 🔥 (20210920) 增加[RobustVideoMatting](https://github.com/PeterL1n/RobustVideoMatting) 视频抠图! 通过[*lite::cv::matting::RobustVideoMatting*](#lite.ai.toolkit-matting)调用! 详见[demo](https://github.com/DefTruth/lite.ai.toolkit/blob/main/examples/lite/cv/test_lite_rvm.cpp).

<div align='center'>
  <img src='docs/resources/interviewi.gif' height="100px" width="100px">
  <img src='docs/resources/interview.gif' height="100px" width="100px">  
  <img src='docs/resources/dance3i.gif' height="100px" width="100px">
  <img src='docs/resources/dance3.gif' height="100px" width="100px">
  <img src='docs/resources/teslai.gif' height="100px" width="100px">
  <img src='docs/resources/tesla.gif' height="100px" width="100px">  
  <img src='docs/resources/b5i.gif' height="100px" width="100px">
  <img src='docs/resources/b5.gif' height="100px" width="100px">
</div>


* 🔥 (20210915) 增加[YOLOP](https://github.com/hustvl/YOLOP) 全景🚗感知! 通过[*lite::cv::detection::YOLOP*](#lite.ai.toolkit-object-detection)调用! 详见[demo](https://github.com/DefTruth/lite.ai.toolkit/blob/main/examples/lite/cv/test_lite_yolop.cpp).

<div align='center'>
  <img src='docs/resources/yolop1.png' height="100px" width="200px">
  <img src='docs/resources/yolop1.gif' height="100px" width="200px">
  <img src='docs/resources/yolop2.png' height="100px" width="200px">
  <img src='docs/resources/yolop2.gif' height="100px" width="200px">

</div>   


* ✅ (20210807) 增加[YoloR](https://github.com/WongKinYiu/yolor) ! 通过[*lite::cv::detection::YoloR*](#lite.ai.toolkit-object-detection)调用! 详见[demo](https://github.com/DefTruth/lite.ai.toolkit/blob/main/examples/lite/cv/test_lite_yolor.cpp).
* ✅ (20210731) 增加[RetinaFace-CVPR2020](https://github.com/biubug6/Pytorch_Retinaface) 超轻量级人脸检测, 仅1.6Mb! 详见[demo](https://github.com/DefTruth/lite.ai.toolkit/blob/main/examples/lite/cv/test_lite_retinaface.cpp).
* 🔥 (20210721) 增加[YOLOX](https://github.com/Megvii-BaseDetection/YOLOX)! 通过[*lite::cv::detection::YoloX*](#lite.ai.toolkit-object-detection)调用! 详见[demo](https://github.com/DefTruth/lite.ai.toolkit/blob/main/examples/lite/cv/test_lite_yolox.cpp).


<details>
<summary> 展开更多更新 </summary>  

## 更多更新 !!!

* ✅ (20210815) 增加 [EfficientDet](https://github.com/zylo117/Yet-Another-EfficientDet-Pytorch) 目标检测! 详见 [demo](https://github.com/DefTruth/lite.ai.toolkit/blob/main/examples/lite/cv/test_lite_efficientdet.cpp).
* ✅ (20210808) 增加 [ScaledYoloV4](https://github.com/WongKinYiu/ScaledYOLOv4) 目标检测! 详见 [demo](https://github.com/DefTruth/lite.ai.toolkit/blob/main/examples/lite/cv/test_lite_scaled_yolov4.cpp).
* ✅ (20210807) 增加 [TinyYoloV4VOC](https://github.com/bubbliiiing/yolov4-tiny-pytorch) 目标检测! 详见 [demo](https://github.com/DefTruth/lite.ai.toolkit/blob/main/examples/lite/cv/test_lite_tiny_yolov4_voc.cpp).
* ✅ (20210807) 增加 [TinyYoloV4COCO](https://github.com/bubbliiiing/yolov4-tiny-pytorch) 目标检测! 详见 [demo](https://github.com/DefTruth/lite.ai.toolkit/blob/main/examples/lite/cv/test_lite_tiny_yolov4_coco.cpp).
* ✅ (20210722) 更新 [lite.ai.toolkit.hub.onnx.md](https://github.com/DefTruth/lite.ai.toolkit/tree/main/docs/hub/lite.ai.toolkit.hub.onnx.md) ! *Lite.AI.Toolkit* 目前包含 *[70+](https://github.com/DefTruth/lite.ai.toolkit/tree/main/docs/hub/lite.ai.toolkit.hub.onnx.md)* AI 模型以及 *[150+](https://github.com/DefTruth/lite.ai.toolkit/tree/main/docs/hub/lite.ai.toolkit.hub.onnx.md)* .onnx文件.
* ⚠️ (20210802) 增加 GPU兼容CUDAExecutionProvider. 详见 [issue#10](https://github.com/DefTruth/lite.ai.toolkit/issues/10).
* ⚠️ (20210801) 修复 [issue#9](https://github.com/DefTruth/lite.ai.toolkit/issues/9) YOLOX 非矩形推理错误. 详见 [yolox.cpp](https://github.com/DefTruth/lite.ai.toolkit/blob/main/ort/cv/yolox.cpp).
* ✅ (20210801) 增加 [FaceBoxes](https://github.com/zisianw/FaceBoxes.PyTorch) 人脸检测! 详见 [demo](https://github.com/DefTruth/lite.ai.toolkit/blob/main/examples/lite/cv/test_lite_faceboxes.cpp).
* ✅ (20210727) 增加 [MobileNetV2SE68、PFLD68](https://github.com/cunjian/pytorch_face_landmark) 人脸68关键点检测! 详见 [demo](https://github.com/DefTruth/lite.ai.toolkit/blob/main/examples/lite/cv/test_lite_pfld68.cpp).
* ✅ (20210726) 增加 [PFLD98](https://github.com/polarisZhao/PFLD-pytorch) 人脸98关键点检测! 详见 [demo](https://github.com/DefTruth/lite.ai.toolkit/blob/main/examples/lite/cv/test_lite_pfld98.cpp).

</details>


## 目录
* [编译](#lite.ai.toolkit-Build-Lite.AI.ToolKit)
* [模型下载](#lite.ai.toolkit-Model-Zoo)
* [应用案例](#lite.ai.toolkit-Examples-for-Lite.AI.ToolKit)
* [API文档](#lite.ai.toolkit-Lite.AI.ToolKit-API-Docs)
* [其他文档](#lite.ai.toolkit-Other-Docs)
* [开源协议](#lite.ai.toolkit-License)
* [引用参考](#lite.ai.toolkit-References)


## 1. 编译Lite.AI.ToolKit

<div id="lite.ai.toolkit-Build-Lite.AI.ToolKit"></div>

从*Lite.AI.ToolKit* 源码编译*MacOS*下的动态库。需要注意的是*Lite.AI.ToolKit* 使用`onnxruntime`作为默认的后端，因为onnxruntime支持大部分onnx的原生算子，具有更高的易用性。


<details>
<summary> Linux 和 Windows </summary>  

### Linux 和 Windows

⚠️ *Lite.AI.ToolKit* 的发行版本目前不直接支持Linux和Windows，你需要从下载*Lite.AI.ToolKit*的源码进行构建。首先，你需要下载(如果有官方编译好的发行版本的话)或编译*OpenCV* and *ONNXRuntime*的动态库，然后把它们放入*third_party*文件夹。请参考依赖库的编译文档[<sup>1</sup>](#lite.ai.toolkit-1)。


* Windows: 你可以参考[issue#6](https://github.com/DefTruth/lite.ai.toolkit/issues/6) ，讨论了常见的编译问题。
* Linux: 参考MacOS下的编译，替换Linux版本的依赖库即可。Linux下的发行版本将会在近期添加 ~ [issue#2](https://github.com/DefTruth/lite.ai.toolkit/issues/2)
* 令人开心的消息!!! : 🚀 你可以直接下载最新的*ONNXRuntime*官方构建的动态库，包含Windows, Linux, MacOS and Arm的版本!!! CPU和GPU的版本均可获得。不需要再从源码进行编译了，nice。可以从[v1.8.1](https://github.com/microsoft/onnxruntime/releases) 下载最新的动态库. 我目前在*Lite.AI.ToolKit*中用的是1.7.0，你可以从[v1.7.0](https://github.com/microsoft/onnxruntime/releases/tag/v1.7.0) 下载, 但1.8.1应该也是可行的。对于*OpenCV*，请尝试从源码构建(Linux) 或者 直接从[OpenCV 4.5.3](https://github.com/opencv/opencv/releases) 下载官方编译好的动态库(Windows). 然后把头文件和依赖库放入 *third_party* 文件夹.

</details>  


* 下载Lite.AI.ToolKit源码:
```shell
git clone --depth=1 https://github.com/DefTruth/lite.ai.toolkit.git  # latest
```


* 编译动态库.
```shell
cd lite.ai.toolkit
sh ./build.sh
```

* GPU兼容性: 详见[issue#10](https://github.com/DefTruth/lite.ai.toolkit/issues/10).

<details>
<summary> 如何链接Lite.AI.ToolKit动态库?</summary>  

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

* 为了链接`lite.ai.toolkit`动态库，你需要确保`OpenCV` and `onnxruntime`也被正确地链接。比如:

```cmake
cmake_minimum_required(VERSION 3.17)
project(testlite.ai.toolkit)
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_BUILD_TYPE debug)
# link opencv.
set(OpenCV_DIR ${CMAKE_SOURCE_DIR}/opencv/lib/cmake/opencv4)
find_package(OpenCV 4 REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})
# link onnxruntime.
set(ONNXRUNTIME_DIR ${CMAKE_SOURCE_DIR}/onnxruntime/)
set(ONNXRUNTIME_INCLUDE_DIR ${ONNXRUNTIME_DIR}/include)
set(ONNXRUNTIME_LIBRARY_DIR ${ONNXRUNTIME_DIR}/lib)
include_directories(${ONNXRUNTIME_INCLUDE_DIR})
link_directories(${ONNXRUNTIME_LIBRARY_DIR})
# link lite.ai.toolkit.
set(LITEHUB_DIR ${CMAKE_SOURCE_DIR}/lite.ai.toolkit)
set(LITEHUB_INCLUDE_DIR ${LITEHUB_DIR}/include)
set(LITEHUB_LIBRARY_DIR ${LITEHUB_DIR}/lib)
include_directories(${LITEHUB_INCLUDE_DIR})
link_directories(${LITEHUB_LIBRARY_DIR})
# add your executable
add_executable(lite_yolov5 test_lite_yolov5.cpp)
target_link_libraries(lite_yolov5 lite.ai.toolkit onnxruntime ${OpenCV_LIBS})
```
你可以在[lite.ai.toolkit.demo](https://github.com/DefTruth/lite.ai.toolkit.demo) 中找到一个简单且完整的，关于如何正确地链接Lite.AI.ToolKit动态库的应用案例。

</details>


## 2. 模型下载  

<div id="lite.ai.toolkit-Model-Zoo"></div>

*Lite.AI.ToolKit* 目前包括 *[70+](https://github.com/DefTruth/lite.ai.toolkit/tree/main/docs/hub/lite.ai.toolkit.hub.onnx.md)* 流行的开源模型以及 *[150+](https://github.com/DefTruth/lite.ai.toolkit/tree/main/docs/hub/lite.ai.toolkit.hub.onnx.md)* .onnx 文件，大部分onnx文件是我自己转换的。你可以通过*lite::cv::Type::Class* 语法进行调用，如 *[lite::cv::detection::YoloV5](#lite.ai.toolkit-object-detection)*。更多的细节见[Examples for Lite.AI.ToolKit](#lite.ai.toolkit-Examples-for-Lite.AI.ToolKit)。

<details>
<summary> 命名空间和Lite.AI.ToolKit算法模块的对应关系 </summary>  

### 命名空间和Lite.AI.ToolKit算法模块的对应关系  

| Namepace                   | Details                                                      |
| :------------------------- | :----------------------------------------------------------- |
| *lite::cv::detection*      | Object Detection. one-stage and anchor-free detectors, YoloV5, YoloV4, SSD, etc. ✅ |
| *lite::cv::classification* | Image Classification. DensNet, ShuffleNet, ResNet, IBNNet, GhostNet, etc. ✅ |
| *lite::cv::faceid*         | Face Recognition. ArcFace, CosFace, CurricularFace, etc. ❇️   |
| *lite::cv::face*           | Face Analysis. *detect*, *align*, *pose*, *attr*, etc. ❇️    |
| *lite::cv::face::detect*   | Face Detection. UltraFace, RetinaFace, FaceBoxes, PyramidBox, etc. ❇️ |
| *lite::cv::face::align*    | Face Alignment. PFLD(106), FaceLandmark1000(1000 landmarks), PRNet, etc. ❇️ |
| *lite::cv::face::pose*     | Head Pose Estimation.  FSANet, etc. ❇️                        |
| *lite::cv::face::attr*     | Face Attributes. Emotion, Age, Gender. EmotionFerPlus, VGG16Age, etc. ❇️ |
| *lite::cv::segmentation*   | Object Segmentation. Such as FCN, DeepLabV3, etc. ⚠️          |
| *lite::cv::style*          | Style Transfer. Contains neural style transfer now, such as FastStyleTransfer.  ⚠️ |
| *lite::cv::matting*        | Image Matting. Object and Human matting.  ⚠️                  |
| *lite::cv::colorization*   | Colorization. Make Gray image become RGB. ⚠️                  |
| *lite::cv::resolution*     | Super Resolution.  ⚠️                                         |


### Lite.AI.ToolKit的类与权重文件对应关系说明

Lite.AI.ToolKit的类与权重文件对应关系说明，可以在[lite.ai.toolkit.hub.onnx.md](https://github.com/DefTruth/lite.ai.toolkit/tree/main/docs/hub/lite.ai.toolkit.hub.onnx.md) 中找到。比如, *lite::cv::detection::YoloV5* 和 *lite::cv::detection::YoloX* 的权重文件为：


|             Class             | Pretrained ONNX Files |                   Rename or Converted From (Repo)                   | Size  |
| :---------------------------: | :-------------------: | :----------------------------------------------------: | :---: |
| *lite::cv::detection::YoloV5* |     yolov5l.onnx      | [yolov5](https://github.com/ultralytics/yolov5) (🔥🔥💥↑) | 188Mb |
| *lite::cv::detection::YoloV5* |     yolov5m.onnx      | [yolov5](https://github.com/ultralytics/yolov5) (🔥🔥💥↑) | 85Mb  |
| *lite::cv::detection::YoloV5* |     yolov5s.onnx      | [yolov5](https://github.com/ultralytics/yolov5) (🔥🔥💥↑) | 29Mb  |
| *lite::cv::detection::YoloV5* |     yolov5x.onnx      | [yolov5](https://github.com/ultralytics/yolov5) (🔥🔥💥↑) | 351Mb |
| *lite::cv::detection::YoloX* |     yolox_x.onnx      | [YOLOX](https://github.com/Megvii-BaseDetection/YOLOX) (🔥🔥!!↑) | 378Mb |
| *lite::cv::detection::YoloX* |     yolox_l.onnx      | [YOLOX](https://github.com/Megvii-BaseDetection/YOLOX) (🔥🔥!!↑) | 207Mb  |
| *lite::cv::detection::YoloX* |     yolox_m.onnx      | [YOLOX](https://github.com/Megvii-BaseDetection/YOLOX) (🔥🔥!!↑) | 97Mb  |
| *lite::cv::detection::YoloX* |     yolox_s.onnx      | [YOLOX](https://github.com/Megvii-BaseDetection/YOLOX) (🔥🔥!!↑) | 34Mb |
| *lite::cv::detection::YoloX* |     yolox_tiny.onnx      | [YOLOX](https://github.com/Megvii-BaseDetection/YOLOX) (🔥🔥!!↑) | 19Mb |
| *lite::cv::detection::YoloX* |     yolox_nano.onnx      | [YOLOX](https://github.com/Megvii-BaseDetection/YOLOX) (🔥🔥!!↑) | 3.5Mb |

这意味着，你可以通过Lite.AI.ToolKit中的同一个类，根据你的使用情况，加载任意一个`yolov5*.onnx`或`yolox_*.onnx`，如 *YoloV5*, *YoloX*等.

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

* 下载链接:  
  [Baidu Drive](https://pan.baidu.com/s/1elUGcx7CZkkjEoYhTMwTRQ) code: 8gin && [Google Drive](https://drive.google.com/drive/folders/1p6uBcxGeyS1exc-T61vL8YRhwjYL4iD2?usp=sharing) .   
  注意，由于Google Driver(15G)的存储限制，我无法上传所有的模型文件，国内的小伙伴请使用百度云盘。<div id="lite.ai.toolkit-2"></div>


* 目标检测

|Class|Size|From|Awesome|File|Type|State|Usage|
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
|[YoloV5](https://github.com/ultralytics/yolov5)|28M|[yolov5](https://github.com/ultralytics/yolov5)|🔥🔥💥↑| [![](https://img.shields.io/badge/onnx-done-brightgreen.svg)](https://github.com/DefTruth/lite.ai.toolkit/tree/main/docs/hub/lite.ai.toolkit.hub.onnx.md#lite.ai.toolkit.hub.onnx-object-detection)  | *detection* | ✅ | [demo](https://github.com/DefTruth/lite.ai.toolkit/blob/main/examples/lite/cv/test_lite_yolov5.cpp) |
|[YoloV3](https://github.com/onnx/models/blob/master/vision/object_detection_segmentation/yolov3)|236M|[onnx-models](https://github.com/onnx/models)|🔥🔥🔥↑| [![](https://img.shields.io/badge/onnx-done-brightgreen.svg)](https://github.com/DefTruth/lite.ai.toolkit/tree/main/docs/hub/lite.ai.toolkit.hub.onnx.md#lite.ai.toolkit.hub.onnx-object-detection) | *detection* | ✅ | [demo](https://github.com/DefTruth/lite.ai.toolkit/blob/main/examples/lite/cv/test_lite_yolov3.cpp) |
|[TinyYoloV3](https://github.com/onnx/models/blob/master/vision/object_detection_segmentation/tiny-yolov3)|33M|        [onnx-models](https://github.com/onnx/models)         | 🔥🔥🔥↑ | [![](https://img.shields.io/badge/onnx-done-brightgreen.svg)](https://github.com/DefTruth/lite.ai.toolkit/tree/main/docs/hub/lite.ai.toolkit.hub.onnx.md#lite.ai.toolkit.hub.onnx-object-detection) | *detection* | ✅ | [demo](https://github.com/DefTruth/lite.ai.toolkit/blob/main/examples/lite/cv/test_lite_tiny_yolov3.cpp) |
|[YoloV4](https://github.com/argusswift/YOLOv4-pytorch)|176M| [YOLOv4...](https://github.com/argusswift/YOLOv4-pytorch) | 🔥🔥🔥↑ | [![](https://img.shields.io/badge/onnx-done-brightgreen.svg)](https://github.com/DefTruth/lite.ai.toolkit/tree/main/docs/hub/lite.ai.toolkit.hub.onnx.md#lite.ai.toolkit.hub.onnx-object-detection) | *detection* | ✅ | [demo](https://github.com/DefTruth/lite.ai.toolkit/blob/main/examples/lite/cv/test_lite_yolov4.cpp) |
|[SSD](https://github.com/onnx/models/blob/master/vision/object_detection_segmentation/ssd)|76M|        [onnx-models](https://github.com/onnx/models)         | 🔥🔥🔥↑ | [![](https://img.shields.io/badge/onnx-done-brightgreen.svg)](https://github.com/DefTruth/lite.ai.toolkit/tree/main/docs/hub/lite.ai.toolkit.hub.onnx.md#lite.ai.toolkit.hub.onnx-object-detection) | *detection* | ✅ | [demo](https://github.com/DefTruth/lite.ai.toolkit/blob/main/examples/lite/cv/test_lite_ssd.cpp) |
|[SSDMobileNetV1](https://github.com/onnx/models/blob/master/vision/object_detection_segmentation/ssd-mobilenetv1)|27M|        [onnx-models](https://github.com/onnx/models)         | 🔥🔥🔥↑ | [![](https://img.shields.io/badge/onnx-done-brightgreen.svg)](https://github.com/DefTruth/lite.ai.toolkit/tree/main/docs/hub/lite.ai.toolkit.hub.onnx.md#lite.ai.toolkit.hub.onnx-object-detection) | *detection* | ✅ | [demo](https://github.com/DefTruth/lite.ai.toolkit/blob/main/examples/lite/cv/test_lite_ssd_mobilenetv1.cpp) |
|[YoloX](https://github.com/Megvii-BaseDetection/YOLOX)|3.5M| [YOLOX](https://github.com/Megvii-BaseDetection/YOLOX) | 🔥🔥new↑ | [![](https://img.shields.io/badge/onnx-done-brightgreen.svg)](https://github.com/DefTruth/lite.ai.toolkit/tree/main/docs/hub/lite.ai.toolkit.hub.onnx.md#lite.ai.toolkit.hub.onnx-object-detection) | *detection* | ✅ | [demo](https://github.com/DefTruth/lite.ai.toolkit/blob/main/examples/lite/cv/test_lite_yolox.cpp) |
|[TinyYoloV4VOC](https://github.com/bubbliiiing/yolov4-tiny-pytorch)|22M| [yolov4-tiny...](https://github.com/bubbliiiing/yolov4-tiny-pytorch) | 🔥🔥↑ | [![](https://img.shields.io/badge/onnx-done-brightgreen.svg)](https://github.com/DefTruth/lite.ai.toolkit.toolkit/tree/main/docs/hub/lite.ai.toolkit.hub.onnx.md#lite.ai.toolkit.hub.onnx-object-detection) | *detection* | ✅ | [demo](https://github.com/DefTruth/lite.ai.toolkit/blob/main/examples/lite/cv/test_lite_tiny_yolov4_voc.cpp) |
|[TinyYoloV4COCO](https://github.com/bubbliiiing/yolov4-tiny-pytorch)|22M| [yolov4-tiny...](https://github.com/bubbliiiing/yolov4-tiny-pytorch) | 🔥🔥↑ | [![](https://img.shields.io/badge/onnx-done-brightgreen.svg)](https://github.com/DefTruth/lite.ai.toolkit/tree/main/docs/hub/lite.ai.toolkit.hub.onnx.md#lite.ai.toolkit.hub.onnx-object-detection) | *detection* | ✅ | [demo](https://github.com/DefTruth/lite.ai.toolkit/blob/main/examples/lite/cv/test_lite_tiny_yolov4_coco.cpp) |
|[YoloR](https://github.com/WongKinYiu/yolor)|39M| [yolor](https://github.com/WongKinYiu/yolor) | 🔥🔥new↑ | [![](https://img.shields.io/badge/onnx-done-brightgreen.svg)](https://github.com/DefTruth/lite.ai.toolkit/tree/main/docs/hub/lite.ai.toolkit.hub.onnx.md#lite.ai.toolkit.hub.onnx-object-detection) | *detection* | ✅ | [demo](https://github.com/DefTruth/lite.ai.toolkit/blob/main/examples/lite/cv/test_lite_yolor.cpp) |
|[ScaledYoloV4](https://github.com/WongKinYiu/ScaledYOLOv4)|270M| [ScaledYOLOv4](https://github.com/WongKinYiu/ScaledYOLOv4) | 🔥🔥🔥↑ | [![](https://img.shields.io/badge/onnx-done-brightgreen.svg)](https://github.com/DefTruth/lite.ai.toolkit/tree/main/docs/hub/lite.ai.toolkit.hub.onnx.md#lite.ai.toolkit.hub.onnx-object-detection) | *detection* | ✅ | [demo](https://github.com/DefTruth/lite.ai.toolkit/blob/main/examples/lite/cv/test_lite_scaled_yolov4.cpp) |
|[EfficientDet](https://github.com/zylo117/Yet-Another-EfficientDet-Pytorch)|15M| [...EfficientDet...](https://github.com/zylo117/Yet-Another-EfficientDet-Pytorch) | 🔥🔥🔥↑ | [![](https://img.shields.io/badge/onnx-done-brightgreen.svg)](https://github.com/DefTruth/lite.ai.toolkit/tree/main/docs/hub/lite.ai.toolkit.hub.onnx.md#lite.ai.toolkit.hub.onnx-object-detection) | *detection* | ✅ | [demo](https://github.com/DefTruth/lite.ai.toolkit/blob/main/examples/lite/cv/test_lite_efficientdet.cpp) |
|[EfficientDetD7](https://github.com/zylo117/Yet-Another-EfficientDet-Pytorch)|220M| [...EfficientDet...](https://github.com/zylo117/Yet-Another-EfficientDet-Pytorch) | 🔥🔥🔥↑ | [![](https://img.shields.io/badge/onnx-done-brightgreen.svg)](https://github.com/DefTruth/lite.ai.toolkit/tree/main/docs/hub/lite.ai.toolkit.hub.onnx.md#lite.ai.toolkit.hub.onnx-object-detection) | *detection* | ✅ | [demo](https://github.com/DefTruth/lite.ai.toolkit/blob/main/examples/lite/cv/test_lite_efficientdet_d7.cpp) |
|[EfficientDetD8](https://github.com/zylo117/Yet-Another-EfficientDet-Pytorch)|322M| [...EfficientDet...](https://github.com/zylo117/Yet-Another-EfficientDet-Pytorch) | 🔥🔥🔥↑ | [![](https://img.shields.io/badge/onnx-done-brightgreen.svg)](https://github.com/DefTruth/lite.ai.toolkit/tree/main/docs/hub/lite.ai.toolkit.hub.onnx.md#lite.ai.toolkit.hub.onnx-object-detection) | *detection* | ✅ | [demo](https://github.com/DefTruth/lite.ai.toolkit/blob/main/examples/lite/cv/test_lite_efficientdet_d8.cpp) |
|[YOLOP](https://github.com/hustvl/YOLOP)|30M| [YOLOP](https://github.com/hustvl/YOLOP) | 🔥🔥new↑ | [![](https://img.shields.io/badge/onnx-done-brightgreen.svg)](https://github.com/DefTruth/lite.ai.toolkit/tree/main/docs/hub/lite.ai.toolkit.hub.onnx.md#lite.ai.toolkit.hub.onnx-object-detection) | *detection* | ✅ | [demo](https://github.com/DefTruth/lite.ai.toolkit/blob/main/examples/lite/cv/test_lite_yolop.cpp) |


* 人脸识别

|Class|Size|From|Awesome|File|Type|State|Usage|
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
|[GlintArcFace](https://github.com/deepinsight/insightface/tree/master/recognition/arcface_torch)|92M|  [insightface](https://github.com/deepinsight/insightface)   | 🔥🔥🔥↑ | [![](https://img.shields.io/badge/onnx-done-brightgreen.svg)](https://github.com/DefTruth/lite.ai.toolkit/tree/main/docs/hub/lite.ai.toolkit.hub.onnx.md#lite.ai.toolkit.hub.onnx-face-recognition) | *faceid* | ✅ | [demo](https://github.com/DefTruth/lite.ai.toolkit/blob/main/examples/lite/cv/test_lite_glint_arcface.cpp) |
|[GlintCosFace](https://github.com/deepinsight/insightface/tree/master/recognition/arcface_torch)|92M|  [insightface](https://github.com/deepinsight/insightface)   | 🔥🔥🔥↑ | [![](https://img.shields.io/badge/onnx-done-brightgreen.svg)](https://github.com/DefTruth/lite.ai.toolkit/tree/main/docs/hub/lite.ai.toolkit.hub.onnx.md#lite.ai.toolkit.hub.onnx-face-recognition) | *faceid* | ✅ | [demo](https://github.com/DefTruth/lite.ai.toolkit/blob/main/examples/lite/cv/test_lite_glint_cosface.cpp) |
|[GlintPartialFC](https://github.com/deepinsight/insightface/tree/master/recognition/partial_fc)|170M|  [insightface](https://github.com/deepinsight/insightface)   | 🔥🔥🔥↑ | [![](https://img.shields.io/badge/onnx-done-brightgreen.svg)](https://github.com/DefTruth/lite.ai.toolkit/tree/main/docs/hub/lite.ai.toolkit.hub.onnx.md#lite.ai.toolkit.hub.onnx-face-recognition) | *faceid* | ✅ | [demo](https://github.com/DefTruth/lite.ai.toolkit/blob/main/examples/lite/cv/test_lite_glint_partial_fc.cpp) |
|[FaceNet](https://github.com/timesler/facenet-pytorch)|89M| [facenet...](https://github.com/timesler/facenet-pytorch) | 🔥🔥🔥↑ | [![](https://img.shields.io/badge/onnx-done-brightgreen.svg)](https://github.com/DefTruth/lite.ai.toolkit/tree/main/docs/hub/lite.ai.toolkit.hub.onnx.md#lite.ai.toolkit.hub.onnx-face-recognition) | *faceid* | ✅ | [demo](https://github.com/DefTruth/lite.ai.toolkit/blob/main/examples/lite/cv/test_lite_facenet.cpp) |
|[FocalArcFace](https://github.com/ZhaoJ9014/face.evoLVe.PyTorch)|166M| [face.evoLVe...](https://github.com/ZhaoJ9014/face.evoLVe.PyTorch) | 🔥🔥🔥↑ | [![](https://img.shields.io/badge/onnx-done-brightgreen.svg)](https://github.com/DefTruth/lite.ai.toolkit/tree/main/docs/hub/lite.ai.toolkit.hub.onnx.md#lite.ai.toolkit.hub.onnx-face-recognition) | *faceid* | ✅ | [demo](https://github.com/DefTruth/lite.ai.toolkit/blob/main/examples/lite/cv/test_lite_focal_arcface.cpp) |
|[FocalAsiaArcFace](https://github.com/ZhaoJ9014/face.evoLVe.PyTorch)|166M| [face.evoLVe...](https://github.com/ZhaoJ9014/face.evoLVe.PyTorch) | 🔥🔥🔥↑ | [![](https://img.shields.io/badge/onnx-done-brightgreen.svg)](https://github.com/DefTruth/lite.ai.toolkit/tree/main/docs/hub/lite.ai.toolkit.hub.onnx.md#lite.ai.toolkit.hub.onnx-face-recognition) | *faceid* | ✅ | [demo](https://github.com/DefTruth/lite.ai.toolkit/blob/main/examples/lite/cv/test_lite_focal_asia_arcface.cpp) |
|[TencentCurricularFace](https://github.com/Tencent/TFace/tree/master/tasks/distfc)|249M|          [TFace](https://github.com/Tencent/TFace)           |  🔥🔥↑  | [![](https://img.shields.io/badge/onnx-done-brightgreen.svg)](https://github.com/DefTruth/lite.ai.toolkit/tree/main/docs/hub/lite.ai.toolkit.hub.onnx.md#lite.ai.toolkit.hub.onnx-face-recognition) | *faceid* | ✅ | [demo](https://github.com/DefTruth/lite.ai.toolkit/blob/main/examples/lite/cv/test_lite_tencent_curricular_face.cpp) |
|[TencentCifpFace](https://github.com/Tencent/TFace/tree/master/tasks/cifp)|130M|          [TFace](https://github.com/Tencent/TFace)           |  🔥🔥↑  | [![](https://img.shields.io/badge/onnx-done-brightgreen.svg)](https://github.com/DefTruth/lite.ai.toolkit/tree/main/docs/hub/lite.ai.toolkit.hub.onnx.md#lite.ai.toolkit.hub.onnx-face-recognition) | *faceid* | ✅ | [demo](https://github.com/DefTruth/lite.ai.toolkit/blob/main/examples/lite/cv/test_lite_tencent_cifp_face.cpp) |
|[CenterLossFace](https://github.com/louis-she/center-loss.pytorch)| 280M |  [center-loss...](https://github.com/louis-she/center-loss.pytorch)           |  🔥🔥↑  | [![](https://img.shields.io/badge/onnx-done-brightgreen.svg)](https://github.com/DefTruth/lite.ai.toolkit/tree/main/docs/hub/lite.ai.toolkit.hub.onnx.md#lite.ai.toolkit.hub.onnx-face-recognition) | *faceid* | ✅ | [demo](https://github.com/DefTruth/lite.ai.toolkit/blob/main/examples/lite/cv/test_lite_center_loss_face.cpp) |
|[SphereFace](https://github.com/clcarwin/sphereface_pytorch)| 80M |  [sphere...](https://github.com/clcarwin/sphereface_pytorch)   |  🔥🔥↑  | [![](https://img.shields.io/badge/onnx-done-brightgreen.svg)](https://github.com/DefTruth/lite.ai.toolkit/tree/main/docs/hub/lite.ai.toolkit.hub.onnx.md#lite.ai.toolkit.hub.onnx-face-recognition) | *faceid* | ✅️ | [demo](https://github.com/DefTruth/lite.ai.toolkit/blob/main/examples/lite/cv/test_lite_sphere_face.cpp) |
|[PoseRobustFace](https://github.com/penincillin/DREAM)| 92M | [DREAM](https://github.com/penincillin/DREAM)  |  🔥🔥↑  | [![](https://img.shields.io/badge/onnx-done-brightgreen.svg)](https://github.com/DefTruth/lite.ai.toolkit/tree/main/docs/hub/lite.ai.toolkit.hub.onnx.md#lite.ai.toolkit.hub.onnx-face-recognition) | *faceid* | ✅️ | [demo](https://github.com/DefTruth/lite.ai.toolkit/blob/main/examples/lite/cv/test_lite_pose_robust_face.cpp) |
|[NaivePoseRobustFace](https://github.com/penincillin/DREAM)| 43M | [DREAM](https://github.com/penincillin/DREAM)  |  🔥🔥↑  | [![](https://img.shields.io/badge/onnx-done-brightgreen.svg)](https://github.com/DefTruth/lite.ai.toolkit/tree/main/docs/hub/lite.ai.toolkit.hub.onnx.md#lite.ai.toolkit.hub.onnx-face-recognition) | *faceid* | ✅️ | [demo](https://github.com/DefTruth/lite.ai.toolkit/blob/main/examples/lite/cv/test_lite_naive_pose_robust_face.cpp) |
|[MobileFaceNet](https://github.com/Xiaoccer/MobileFaceNet_Pytorch)| 3.8M |  [MobileFace...](https://github.com/Xiaoccer/MobileFaceNet_Pytorch)           |  🔥🔥↑  | [![](https://img.shields.io/badge/onnx-done-brightgreen.svg)](https://github.com/DefTruth/lite.ai.toolkit/tree/main/docs/hub/lite.ai.toolkit.hub.onnx.md#lite.ai.toolkit.hub.onnx-face-recognition) | *faceid* | ✅ | [demo](https://github.com/DefTruth/lite.ai.toolkit/blob/main/examples/lite/cv/test_lite_mobile_facenet.cpp) |
|[CavaGhostArcFace](https://github.com/cavalleria/cavaface.pytorch)| 15M | [cavaface...](https://github.com/cavalleria/cavaface.pytorch) |  🔥🔥↑  | [![](https://img.shields.io/badge/onnx-done-brightgreen.svg)](https://github.com/DefTruth/lite.ai.toolkit/tree/main/docs/hub/lite.ai.toolkit.hub.onnx.md#lite.ai.toolkit.hub.onnx-face-recognition) | *faceid* |  ✅ | [demo](https://github.com/DefTruth/lite.ai.toolkit/blob/main/examples/lite/cv/test_lite_cava_ghost_arcface.cpp) |
|[CavaCombinedFace](https://github.com/cavalleria/cavaface.pytorch)| 250M | [cavaface...](https://github.com/cavalleria/cavaface.pytorch) |  🔥🔥↑  | [![](https://img.shields.io/badge/onnx-done-brightgreen.svg)](https://github.com/DefTruth/lite.ai.toolkit/tree/main/docs/hub/lite.ai.toolkit.hub.onnx.md#lite.ai.toolkit.hub.onnx-face-recognition) | *faceid* | ✅ | [demo](https://github.com/DefTruth/lite.ai.toolkit/blob/main/examples/lite/cv/test_lite_cava_combined_face.cpp) |
|[MobileSEFocalFace](https://github.com/grib0ed0v/face_recognition.pytorch)|4.5M| [face_recog...](https://github.com/grib0ed0v/face_recognition.pytorch) | 🔥🔥↑ | [![](https://img.shields.io/badge/onnx-done-brightgreen.svg)](https://github.com/DefTruth/lite.ai.toolkit/tree/main/docs/hub/lite.ai.toolkit.hub.onnx.md#lite.ai.toolkit.hub.onnx-face-recognition) | *faceid* | ✅ |  [demo](https://github.com/DefTruth/lite.ai.toolkit/blob/main/examples/lite/cv/test_lite_mobilese_focal_face.cpp) |

* 抠图

|Class|Size|From|Awesome|File|Type|State|Usage|
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
|[RobustVideoMatting](https://github.com/PeterL1n/RobustVideoMatting)|14M| [RobustVideoMatting](https://github.com/PeterL1n/RobustVideoMatting)  |   🔥🔥🔥latest↑   | [![](https://img.shields.io/badge/onnx-done-brightgreen.svg)](https://github.com/DefTruth/lite.ai.toolkit/tree/main/docs/hub/lite.ai.toolkit.hub.onnx.md#lite.ai.toolkit.hub.onnx-matting) | *matting* | ✅ | [demo](https://github.com/DefTruth/lite.ai.toolkit/blob/main/examples/lite/cv/test_lite_rvm.cpp) |


<details>
<summary> ⚠️ 展开Lite.AI.ToolKit的Model Zoo中更多的模型 </summary>  

* 人脸检测

|Class|Size|From|Awesome|File|Type|State|Usage|
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
|[UltraFace](https://github.com/Linzaer/Ultra-Light-Fast-Generic-Face-Detector-1MB)|1.1M| [Ultra-Light...](https://github.com/Linzaer/Ultra-Light-Fast-Generic-Face-Detector-1MB) | 🔥🔥🔥↑ | [![](https://img.shields.io/badge/onnx-done-brightgreen.svg)](https://github.com/DefTruth/lite.ai.toolkit/tree/main/docs/hub/lite.ai.toolkit.hub.onnx.md#lite.ai.toolkit.hub.onnx-face-detection) | *face::detect* | ✅ | [demo](https://github.com/DefTruth/lite.ai.toolkit/blob/main/examples/lite/cv/test_lite_ultraface.cpp) |
|[RetinaFace](https://github.com/biubug6/Pytorch_Retinaface)|1.6M| [...Retinaface](https://github.com/biubug6/Pytorch_Retinaface) | 🔥🔥🔥↑ | [![](https://img.shields.io/badge/onnx-done-brightgreen.svg)](https://github.com/DefTruth/lite.ai.toolkit/tree/main/docs/hub/lite.ai.toolkit.hub.onnx.md#lite.ai.toolkit.hub.onnx-face-detection) | *face::detect* | ✅ | [demo](https://github.com/DefTruth/lite.ai.toolkit/blob/main/examples/lite/cv/test_lite_retinaface.cpp) |
|[FaceBoxes](https://github.com/zisianw/FaceBoxes.PyTorch)|3.8M| [FaceBoxes](https://github.com/zisianw/FaceBoxes.PyTorch) | 🔥🔥↑ | [![](https://img.shields.io/badge/onnx-done-brightgreen.svg)](https://github.com/DefTruth/lite.ai.toolkit/tree/main/docs/hub/lite.ai.toolkit.hub.onnx.md#lite.ai.toolkit.hub.onnx-face-detection) | *face::detect* | ✅ | [demo](https://github.com/DefTruth/lite.ai.toolkit/blob/main/examples/lite/cv/test_lite_faceboxes.cpp) |


* 人脸对齐

|Class|Size|From|Awesome|File|Type|State|Usage|
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
|[PFLD](https://github.com/Hsintao/pfld_106_face_landmarks)|1.0M| [pfld_106_...](https://github.com/Hsintao/pfld_106_face_landmarks) |  🔥🔥↑  | [![](https://img.shields.io/badge/onnx-done-brightgreen.svg)](https://github.com/DefTruth/lite.ai.toolkit/tree/main/docs/hub/lite.ai.toolkit.hub.onnx.md#lite.ai.toolkit.hub.onnx-face-alignment) | *face::align* | ✅ | [demo](https://github.com/DefTruth/lite.ai.toolkit/blob/main/examples/lite/cv/test_lite_pfld.cpp) |
|[PFLD98](https://github.com/polarisZhao/PFLD-pytorch)|4.8M| [PFLD...](https://github.com/polarisZhao/PFLD-pytorch) | 🔥🔥↑ | [![](https://img.shields.io/badge/onnx-done-brightgreen.svg)](https://github.com/DefTruth/lite.ai.toolkit/tree/main/docs/hub/lite.ai.toolkit.hub.onnx.md#lite.ai.toolkit.hub.onnx-face-alignment) | *face::align* | ✅️ | [demo](https://github.com/DefTruth/lite.ai.toolkit/blob/main/examples/lite/cv/test_lite_pfld98.cpp) |
|[MobileNetV268](https://github.com/cunjian/pytorch_face_landmark)|9.4M| [...landmark](https://github.com/cunjian/pytorch_face_landmark) | 🔥🔥↑ |  [![](https://img.shields.io/badge/onnx-done-brightgreen.svg)](https://github.com/DefTruth/lite.ai.toolkit/tree/main/docs/hub/lite.ai.toolkit.hub.onnx.md#lite.ai.toolkit.hub.onnx-face-alignment) | *face::align* | ✅️️ | [demo](https://github.com/DefTruth/lite.ai.toolkit/blob/main/examples/lite/cv/test_lite_mobilenetv2_68.cpp) |
|[MobileNetV2SE68](https://github.com/cunjian/pytorch_face_landmark)|11M| [...landmark](https://github.com/cunjian/pytorch_face_landmark) | 🔥🔥↑ |  [![](https://img.shields.io/badge/onnx-done-brightgreen.svg)](https://github.com/DefTruth/lite.ai.toolkit/tree/main/docs/hub/lite.ai.toolkit.hub.onnx.md#lite.ai.toolkit.hub.onnx-face-alignment) | *face::align* | ✅️️ | [demo](https://github.com/DefTruth/lite.ai.toolkit/blob/main/examples/lite/cv/test_lite_mobilenetv2_se_68.cpp) |
|[PFLD68](https://github.com/cunjian/pytorch_face_landmark)|2.8M| [...landmark](https://github.com/cunjian/pytorch_face_landmark) | 🔥🔥↑ |  [![](https://img.shields.io/badge/onnx-done-brightgreen.svg)](https://github.com/DefTruth/lite.ai.toolkit/tree/main/docs/hub/lite.ai.toolkit.hub.onnx.md#lite.ai.toolkit.hub.onnx-face-alignment) | *face::align* | ✅️ | [demo](https://github.com/DefTruth/lite.ai.toolkit/blob/main/examples/lite/cv/test_lite_pfld68.cpp) |
|[FaceLandmark1000](https://github.com/Single430/FaceLandmark1000)|2.0M| [FaceLandm...](https://github.com/Single430/FaceLandmark1000) | 🔥↑ |  [![](https://img.shields.io/badge/onnx-done-brightgreen.svg)](https://github.com/DefTruth/lite.ai.toolkit/tree/main/docs/hub/lite.ai.toolkit.hub.onnx.md#lite.ai.toolkit.hub.onnx-face-alignment) | *face::align* | ✅️ | [demo](https://github.com/DefTruth/lite.ai.toolkit/blob/main/examples/lite/cv/test_lite_face_landmarks_1000.cpp) |



* 头部姿态估计

|Class|Size|From|Awesome|File|Type|State|Usage|
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
|[FSANet](https://github.com/omasaht/headpose-fsanet-pytorch)|1.2M| [...fsanet...](https://github.com/omasaht/headpose-fsanet-pytorch) | 🔥↑ | [![](https://img.shields.io/badge/onnx-done-brightgreen.svg)](https://github.com/DefTruth/lite.ai.toolkit/tree/main/docs/hub/lite.ai.toolkit.hub.onnx.md#lite.ai.toolkit.hub.onnx-head-pose-estimation) | *face::pose* | ✅ | [demo](https://github.com/DefTruth/lite.ai.toolkit/blob/main/examples/lite/cv/test_lite_fsanet.cpp) |


* 人脸属性识别

|Class|Size|From|Awesome|File|Type|State|Usage|
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
|[AgeGoogleNet](https://github.com/onnx/models/tree/master/vision/body_analysis/age_gender)|23M|        [onnx-models](https://github.com/onnx/models)         | 🔥🔥🔥↑ | [![](https://img.shields.io/badge/onnx-done-brightgreen.svg)](https://github.com/DefTruth/lite.ai.toolkit/tree/main/docs/hub/lite.ai.toolkit.hub.onnx.md#lite.ai.toolkit.hub.onnx-face-attributes) | *face::attr* | ✅ | [demo](https://github.com/DefTruth/lite.ai.toolkit/blob/main/examples/lite/cv/test_lite_age_googlenet.cpp) |
|[GenderGoogleNet](https://github.com/onnx/models/tree/master/vision/body_analysis/age_gender)|23M|        [onnx-models](https://github.com/onnx/models)         | 🔥🔥🔥↑ | [![](https://img.shields.io/badge/onnx-done-brightgreen.svg)](https://github.com/DefTruth/lite.ai.toolkit/tree/main/docs/hub/lite.ai.toolkit.hub.onnx.md#lite.ai.toolkit.hub.onnx-face-attributes) | *face::attr* | ✅ | [demo](https://github.com/DefTruth/lite.ai.toolkit/blob/main/examples/lite/cv/test_lite_gender_googlenet.cpp) |
|[EmotionFerPlus](https://github.com/onnx/models/blob/master/vision/body_analysis/emotion_ferplus)|33M|        [onnx-models](https://github.com/onnx/models)         | 🔥🔥🔥↑ | [![](https://img.shields.io/badge/onnx-done-brightgreen.svg)](https://github.com/DefTruth/lite.ai.toolkit/tree/main/docs/hub/lite.ai.toolkit.hub.onnx.md#lite.ai.toolkit.hub.onnx-face-attributes) | *face::attr* | ✅ | [demo](https://github.com/DefTruth/lite.ai.toolkit/blob/main/examples/lite/cv/test_lite_emotion_ferplus.cpp) |
|[VGG16Age](https://github.com/onnx/models/tree/master/vision/body_analysis/age_gender)|514M|        [onnx-models](https://github.com/onnx/models)         | 🔥🔥🔥↑ | [![](https://img.shields.io/badge/onnx-done-brightgreen.svg)](https://github.com/DefTruth/lite.ai.toolkit/tree/main/docs/hub/lite.ai.toolkit.hub.onnx.md#lite.ai.toolkit.hub.onnx-face-attributes) | *face::attr* | ✅ | [demo](https://github.com/DefTruth/lite.ai.toolkit/blob/main/examples/lite/cv/test_lite_vgg16_age.cpp) |
|[VGG16Gender](https://github.com/onnx/models/tree/master/vision/body_analysis/age_gender)|512M|        [onnx-models](https://github.com/onnx/models)         | 🔥🔥🔥↑ | [![](https://img.shields.io/badge/onnx-done-brightgreen.svg)](https://github.com/DefTruth/lite.ai.toolkit/tree/main/docs/hub/lite.ai.toolkit.hub.onnx.md#lite.ai.toolkit.hub.onnx-face-attributes) | *face::attr* | ✅ | [demo](https://github.com/DefTruth/lite.ai.toolkit/blob/main/examples/lite/cv/test_lite_vgg16_gender.cpp) |
|[SSRNet](https://github.com/oukohou/SSR_Net_Pytorch)|190K| [SSR_Net...](https://github.com/oukohou/SSR_Net_Pytorch) | 🔥↑ | [![](https://img.shields.io/badge/onnx-done-brightgreen.svg)](https://github.com/DefTruth/lite.ai.toolkit/tree/main/docs/hub/lite.ai.toolkit.hub.onnx.md#lite.ai.toolkit.hub.onnx-face-attributes) | *face::attr* | ✅ | [demo](https://github.com/DefTruth/lite.ai.toolkit/blob/main/examples/lite/cv/test_lite_ssrnet.cpp) |
|[EfficientEmotion7](https://github.com/HSE-asavchenko/face-emotion-recognition)|15M| [face-emo...](https://github.com/HSE-asavchenko/face-emotion-recognition) | 🔥↑ | [![](https://img.shields.io/badge/onnx-done-brightgreen.svg)](https://github.com/DefTruth/lite.ai.toolkit/tree/main/docs/hub/lite.ai.toolkit.hub.onnx.md#lite.ai.toolkit.hub.onnx-face-attributes) | *face::attr* | ✅️ | [demo](https://github.com/DefTruth/lite.ai.toolkit/blob/main/examples/lite/cv/test_lite_efficient_emotion7.cpp) |
|[EfficientEmotion8](https://github.com/HSE-asavchenko/face-emotion-recognition)|15M| [face-emo...](https://github.com/HSE-asavchenko/face-emotion-recognition) | 🔥↑ | [![](https://img.shields.io/badge/onnx-done-brightgreen.svg)](https://github.com/DefTruth/lite.ai.toolkit/tree/main/docs/hub/lite.ai.toolkit.hub.onnx.md#lite.ai.toolkit.hub.onnx-face-attributes) | *face::attr* | ✅  | [demo](https://github.com/DefTruth/lite.ai.toolkit/blob/main/examples/lite/cv/test_lite_efficient_emotion8.cpp) |
|[MobileEmotion7](https://github.com/HSE-asavchenko/face-emotion-recognition)|13M| [face-emo...](https://github.com/HSE-asavchenko/face-emotion-recognition) | 🔥↑ | [![](https://img.shields.io/badge/onnx-done-brightgreen.svg)](https://github.com/DefTruth/lite.ai.toolkit/tree/main/docs/hub/lite.ai.toolkit.hub.onnx.md#lite.ai.toolkit.hub.onnx-face-attributes) | *face::attr* |  ✅  | [demo](https://github.com/DefTruth/lite.ai.toolkit/blob/main/examples/lite/cv/test_lite_mobile_emotion7.cpp) |
|[ReXNetEmotion7](https://github.com/HSE-asavchenko/face-emotion-recognition)|30M| [face-emo...](https://github.com/HSE-asavchenko/face-emotion-recognition) | 🔥↑ | [![](https://img.shields.io/badge/onnx-done-brightgreen.svg)](https://github.com/DefTruth/lite.ai.toolkit/tree/main/docs/hub/lite.ai.toolkit.hub.onnx.md#lite.ai.toolkit.hub.onnx-face-attributes) | *face::attr* |  ✅  | [demo](https://github.com/DefTruth/lite.ai.toolkit/blob/main/examples/lite/cv/test_lite_rexnet_emotion7.cpp) |


* 图像分类  

|Class|Size|From|Awesome|File|Type|State|Usage|
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
|[EfficientNetLite4](https://github.com/onnx/models/blob/master/vision/classification/efficientnet-lite4)|49M|        [onnx-models](https://github.com/onnx/models)         | 🔥🔥🔥↑ | [![](https://img.shields.io/badge/onnx-done-brightgreen.svg)](https://github.com/DefTruth/lite.ai.toolkit/tree/main/docs/hub/lite.ai.toolkit.hub.onnx.md#lite.ai.toolkit.hub.onnx-classification)  | *classification* | ✅ | [demo](https://github.com/DefTruth/lite.ai.toolkit/blob/main/examples/lite/cv/test_lite_efficientnet_lite4.cpp) |
|[ShuffleNetV2](https://github.com/onnx/models/blob/master/vision/classification/shufflenet)|8.7M|        [onnx-models](https://github.com/onnx/models)         | 🔥🔥🔥↑ | [![](https://img.shields.io/badge/onnx-done-brightgreen.svg)](https://github.com/DefTruth/lite.ai.toolkit/tree/main/docs/hub/lite.ai.toolkit.hub.onnx.md#lite.ai.toolkit.hub.onnx-classification)  | *classification* | ✅ | [demo](https://github.com/DefTruth/lite.ai.toolkit/blob/main/examples/lite/cv/test_lite_shufflenetv2.cpp) |
|[DenseNet121](https://pytorch.org/hub/pytorch_vision_densenet/)|30.7M|       [torchvision](https://github.com/pytorch/vision)       | 🔥🔥🔥↑ | [![](https://img.shields.io/badge/onnx-done-brightgreen.svg)](https://github.com/DefTruth/lite.ai.toolkit/tree/main/docs/hub/lite.ai.toolkit.hub.onnx.md#lite.ai.toolkit.hub.onnx-classification)  | *classification* | ✅ | [demo](https://github.com/DefTruth/lite.ai.toolkit/blob/main/examples/lite/cv/test_lite_densenet.cpp) |
|[GhostNet](https://pytorch.org/hub/pytorch_vision_ghostnet/)|20M|       [torchvision](https://github.com/pytorch/vision)       | 🔥🔥🔥↑ | [![](https://img.shields.io/badge/onnx-done-brightgreen.svg)](https://github.com/DefTruth/lite.ai.toolkit/tree/main/docs/hub/lite.ai.toolkit.hub.onnx.md#lite.ai.toolkit.hub.onnx-classification)  | *classification* | ✅ | [demo](https://github.com/DefTruth/lite.ai.toolkit/blob/main/examples/lite/cv/test_lite_ghostnet.cpp) |
|[HdrDNet](https://pytorch.org/hub/pytorch_vision_hardnet//)|13M|       [torchvision](https://github.com/pytorch/vision)       | 🔥🔥🔥↑ | [![](https://img.shields.io/badge/onnx-done-brightgreen.svg)](https://github.com/DefTruth/lite.ai.toolkit/tree/main/docs/hub/lite.ai.toolkit.hub.onnx.md#lite.ai.toolkit.hub.onnx-classification) | *classification* | ✅ | [demo](https://github.com/DefTruth/lite.ai.toolkit/blob/main/examples/lite/cv/test_lite_hardnet.cpp) |
|[IBNNet](https://pytorch.org/hub/pytorch_vision_ibnnet/)|97M|       [torchvision](https://github.com/pytorch/vision)       | 🔥🔥🔥↑ | [![](https://img.shields.io/badge/onnx-done-brightgreen.svg)](https://github.com/DefTruth/lite.ai.toolkit/tree/main/docs/hub/lite.ai.toolkit.hub.onnx.md#lite.ai.toolkit.hub.onnx-classification)  | *classification* | ✅ | [demo](https://github.com/DefTruth/lite.ai.toolkit/blob/main/examples/lite/cv/test_lite_ibnnet.cpp) |
|[MobileNetV2](https://pytorch.org/hub/pytorch_vision_mobilenet_v2/)|13M|       [torchvision](https://github.com/pytorch/vision)       | 🔥🔥🔥↑ | [![](https://img.shields.io/badge/onnx-done-brightgreen.svg)](https://github.com/DefTruth/lite.ai.toolkit/tree/main/docs/hub/lite.ai.toolkit.hub.onnx.md#lite.ai.toolkit.hub.onnx-classification)  | *classification* | ✅ | [demo](https://github.com/DefTruth/lite.ai.toolkit/blob/main/examples/lite/cv/test_lite_mobilenetv2.cpp) |
|[ResNet](https://pytorch.org/hub/pytorch_vision_resnet/)|44M|       [torchvision](https://github.com/pytorch/vision)       | 🔥🔥🔥↑ | [![](https://img.shields.io/badge/onnx-done-brightgreen.svg)](https://github.com/DefTruth/lite.ai.toolkit/tree/main/docs/hub/lite.ai.toolkit.hub.onnx.md#lite.ai.toolkit.hub.onnx-classification)  | *classification* | ✅ | [demo](https://github.com/DefTruth/lite.ai.toolkit/blob/main/examples/lite/cv/test_lite_resnet.cpp) |
|[ResNeXt](https://pytorch.org/hub/pytorch_vision_resnext/)|95M|       [torchvision](https://github.com/pytorch/vision)       | 🔥🔥🔥↑ | [![](https://img.shields.io/badge/onnx-done-brightgreen.svg)](https://github.com/DefTruth/lite.ai.toolkit/tree/main/docs/hub/lite.ai.toolkit.hub.onnx.md#lite.ai.toolkit.hub.onnx-classification)  | *classification* | ✅ | [demo](https://github.com/DefTruth/lite.ai.toolkit/blob/main/examples/lite/cv/test_lite_resnext.cpp) |


* 语义分割

|Class|Size|From|Awesome|File|Type|State|Usage|
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
|[DeepLabV3ResNet101](https://pytorch.org/hub/pytorch_vision_deeplabv3_resnet101/)|232M|       [torchvision](https://github.com/pytorch/vision)       | 🔥🔥🔥↑ | [![](https://img.shields.io/badge/onnx-done-brightgreen.svg)](https://github.com/DefTruth/lite.ai.toolkit/tree/main/docs/hub/lite.ai.toolkit.hub.onnx.md#lite.ai.toolkit.hub.onnx-segmentation) | *segmentation* | ✅ | [demo](https://github.com/DefTruth/lite.ai.toolkit/blob/main/examples/lite/cv/test_lite_deeplabv3_resnet101.cpp) |
|[FCNResNet101](https://pytorch.org/hub/pytorch_vision_fcn_resnet101/)|207M|       [torchvision](https://github.com/pytorch/vision)       | 🔥🔥🔥↑ | [![](https://img.shields.io/badge/onnx-done-brightgreen.svg)](https://github.com/DefTruth/lite.ai.toolkit/tree/main/docs/hub/lite.ai.toolkit.hub.onnx.md#lite.ai.toolkit.hub.onnx-segmentation) | *segmentation* | ✅ | [demo](https://github.com/DefTruth/lite.ai.toolkit/blob/main/examples/lite/cv/test_lite_fcn_resnet101.cpp) |


* 风格迁移

|Class|Size|From|Awesome|File|Type|State|Usage|
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
|[FastStyleTransfer](https://github.com/onnx/models/blob/master/vision/style_transfer/fast_neural_style)|6.4M|        [onnx-models](https://github.com/onnx/models)         | 🔥🔥🔥↑ | [![](https://img.shields.io/badge/onnx-done-brightgreen.svg)](https://github.com/DefTruth/lite.ai.toolkit/tree/main/docs/hub/lite.ai.toolkit.hub.onnx.md#lite.ai.toolkit.hub.onnx-style-transfer) | *style* | ✅ | [demo](https://github.com/DefTruth/lite.ai.toolkit/blob/main/examples/lite/cv/test_lite_fast_style_transfer.cpp) |


* 图片着色

|Class|Size|From|Awesome|File|Type|State|Usage|
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
|[Colorizer](https://github.com/richzhang/colorization)|123M|  [colorization](https://github.com/richzhang/colorization)   | 🔥🔥🔥↑ | [![](https://img.shields.io/badge/onnx-done-brightgreen.svg)](https://github.com/DefTruth/lite.ai.toolkit/tree/main/docs/hub/lite.ai.toolkit.hub.onnx.md#lite.ai.toolkit.hub.onnx-colorization) | *colorization* | ✅ | [demo](https://github.com/DefTruth/lite.ai.toolkit/blob/main/examples/lite/cv/test_lite_colorizer.cpp) |


* 超分辨率

|Class|Size|From|Awesome|File|Type|State|Usage|
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
|[SubPixelCNN](https://github.com/niazwazir/SUB_PIXEL_CNN)|234K| [...PIXEL...](https://github.com/niazwazir/SUB_PIXEL_CNN)  |    🔥↑    | [![](https://img.shields.io/badge/onnx-done-brightgreen.svg)](https://github.com/DefTruth/lite.ai.toolkit/tree/main/docs/hub/lite.ai.toolkit.hub.onnx.md#lite.ai.toolkit.hub.onnx-super-resolution) | *resolution* | ✅ | [demo](https://github.com/DefTruth/lite.ai.toolkit/blob/main/examples/lite/cv/test_lite_subpixel_cnn.cpp) |

</details>

## 3. 应用案例.

<div id="lite.ai.toolkit-Examples-for-Lite.AI.ToolKit"></div>

更多的应用案例详见[lite.ai.toolkit.examples](https://github.com/DefTruth/lite.ai.toolkit/tree/main/examples/lite/cv) 。点击 ▶️ 可以看到该主题下更多的案例。

<div id="lite.ai.toolkit-object-detection"></div>

#### 案例0: 使用[YoloV5](https://github.com/ultralytics/yolov5) 进行目标检测。请从Model-Zoo[<sup>2</sup>](#lite.ai.toolkit-2) 下载模型文件。
```c++
#include "lite/lite.h"

static void test_default()
{
  std::string onnx_path = "../../../hub/onnx/cv/yolov5s.onnx";
  std::string test_img_path = "../../../examples/lite/resources/test_lite_yolov5_1.jpg";
  std::string save_img_path = "../../../logs/test_lite_yolov5_1.jpg";

  auto *yolov5 = new lite::cv::detection::YoloV5(onnx_path); 
  std::vector<lite::cv::types::Boxf> detected_boxes;
  cv::Mat img_bgr = cv::imread(test_img_path);
  yolov5->detect(img_bgr, detected_boxes);
  
  lite::cv::utils::draw_boxes_inplace(img_bgr, detected_boxes);
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
  std::vector<lite::cv::types::MattingContent> contents;
  
  // 1. video matting.
  rvm->detect_video(video_path, output_path, contents, false, 0.4f);
  
  delete rvm;
}
```
输出的结果是:

<div align='center'>
  <img src='docs/resources/interviewi.gif' height="200px" width="200px">
  <img src='docs/resources/interview.gif' height="200px" width="200px">  
  <img src='docs/resources/dance3i.gif' height="200px" width="200px">
  <img src='docs/resources/dance3.gif' height="200px" width="200px">
  <br>
  <img src='docs/resources/teslai.gif' height="200px" width="200px">
  <img src='docs/resources/tesla.gif' height="200px" width="200px">  
  <img src='docs/resources/b5i.gif' height="200px" width="200px">
  <img src='docs/resources/b5.gif' height="200px" width="200px">
</div>


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

  lite::cv::types::Landmarks landmarks;
  cv::Mat img_bgr = cv::imread(test_img_path);
  face_landmarks_1000->detect(img_bgr, landmarks);
  lite::cv::utils::draw_landmarks_inplace(img_bgr, landmarks);
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
  lite::cv::types::ColorizeContent colorize_content;
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

  lite::cv::types::FaceContent face_content0, face_content1, face_content2;
  cv::Mat img_bgr0 = cv::imread(test_img_path0);
  cv::Mat img_bgr1 = cv::imread(test_img_path1);
  cv::Mat img_bgr2 = cv::imread(test_img_path2);
  glint_arcface->detect(img_bgr0, face_content0);
  glint_arcface->detect(img_bgr1, face_content1);
  glint_arcface->detect(img_bgr2, face_content2);

  if (face_content0.flag && face_content1.flag && face_content2.flag)
  {
    float sim01 = lite::cv::utils::math::cosine_similarity<float>(
        face_content0.embedding, face_content1.embedding);
    float sim02 = lite::cv::utils::math::cosine_similarity<float>(
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

****

<div id="lite.ai.toolkit-face-detection"></div>

#### 案例5: 使用[UltraFace](https://github.com/Linzaer/Ultra-Light-Fast-Generic-Face-Detector-1MB) 进行人脸检测。请从Model-Zoo[<sup>2</sup>](#lite.ai.toolkit-2) 下载模型文件。
```c++
#include "lite/lite.h"

static void test_default()
{
  std::string onnx_path = "../../../hub/onnx/cv/ultraface-rfb-640.onnx";
  std::string test_img_path = "../../../examples/lite/resources/test_lite_ultraface.jpg";
  std::string save_img_path = "../../../logs/test_lite_ultraface.jpg";

  auto *ultraface = new lite::cv::face::detect::UltraFace(onnx_path);

  std::vector<lite::cv::types::Boxf> detected_boxes;
  cv::Mat img_bgr = cv::imread(test_img_path);
  ultraface->detect(img_bgr, detected_boxes);
  lite::cv::utils::draw_boxes_inplace(img_bgr, detected_boxes);
  cv::imwrite(save_img_path, img_bgr);

  delete ultraface;
}
```
输出的结果是:
<div align='center'>
  <img src='logs/test_lite_ultraface.jpg' height="224px" width="224px">
  <img src='logs/test_lite_ultraface_2.jpg' height="224px" width="224px">
  <img src='logs/test_lite_ultraface_3.jpg' height="224px" width="224px">
</div>  


<div id="lite.ai.toolkit-segmentation"></div>  
<div id="lite.ai.toolkit-face-attributes-analysis"></div>  
<div id="lite.ai.toolkit-image-classification"></div>  
<div id="lite.ai.toolkit-head-pose-estimation"></div>
<div id="lite.ai.toolkit-style-transfer"></div>

<details>
<summary> ⚠️ 展开Lite.AI.ToolKit所有算法模块的应用案例 </summary>  

<details>
<summary> 3.1 目标检测应用案例 </summary>

#### 3.1 使用[YoloV5](https://github.com/ultralytics/yolov5) 进行目标检测。请从Model-Zoo[<sup>2</sup>](#lite.ai.toolkit-2) 下载模型文件。
```c++
#include "lite/lite.h"

static void test_default()
{
  std::string onnx_path = "../../../hub/onnx/cv/yolov5s.onnx";
  std::string test_img_path = "../../../examples/lite/resources/test_lite_yolov5_1.jpg";
  std::string save_img_path = "../../../logs/test_lite_yolov5_1.jpg";
  
  auto *yolov5 = new lite::cv::detection::YoloV5(onnx_path);
  std::vector<lite::cv::types::Boxf> detected_boxes;
  cv::Mat img_bgr = cv::imread(test_img_path);
  yolov5->detect(img_bgr, detected_boxes);
  
  lite::cv::utils::draw_boxes_inplace(img_bgr, detected_boxes);
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

```c++
#include "lite/lite.h"

static void test_default()
{
  std::string onnx_path = "../../../hub/onnx/cv/yolox_s.onnx";
  std::string test_img_path = "../../../examples/lite/resources/test_lite_yolox_1.jpg";
  std::string save_img_path = "../../../logs/test_lite_yolox_1.jpg";

  auto *yolox = new lite::cv::detection::YoloX(onnx_path); 
  std::vector<lite::cv::types::Boxf> detected_boxes;
  cv::Mat img_bgr = cv::imread(test_img_path);
  yolox->detect(img_bgr, detected_boxes);
  
  lite::cv::utils::draw_boxes_inplace(img_bgr, detected_boxes);
  cv::imwrite(save_img_path, img_bgr);  
  
  delete yolox;
}
```
输出的结果是:
<div align='center'>
  <img src='logs/test_lite_yolox_1.jpg' height="256px">
  <img src='logs/test_lite_yolox_2.jpg' height="256px">
</div>    

目标检测更多可使用的算法包括：
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
```

</details>


<details>
<summary> 3.2 人脸识别应用案例 </summary>

#### 3.2 使用[ArcFace](https://github.com/deepinsight/insightface/tree/master/recognition/arcface_torch) 进行人脸识别。请从Model-Zoo[<sup>2</sup>](#lite.ai.toolkit-2) 下载模型文件。

```c++
#include "lite/lite.h"

static void test_default()
{
  std::string onnx_path = "../../../hub/onnx/cv/ms1mv3_arcface_r100.onnx";
  std::string test_img_path0 = "../../../examples/lite/resources/test_lite_faceid_0.png";
  std::string test_img_path1 = "../../../examples/lite/resources/test_lite_faceid_1.png";
  std::string test_img_path2 = "../../../examples/lite/resources/test_lite_faceid_2.png";

  auto *glint_arcface = new lite::cv::faceid::GlintArcFace(onnx_path);

  lite::cv::types::FaceContent face_content0, face_content1, face_content2;
  cv::Mat img_bgr0 = cv::imread(test_img_path0);
  cv::Mat img_bgr1 = cv::imread(test_img_path1);
  cv::Mat img_bgr2 = cv::imread(test_img_path2);
  glint_arcface->detect(img_bgr0, face_content0);
  glint_arcface->detect(img_bgr1, face_content1);
  glint_arcface->detect(img_bgr2, face_content2);

  if (face_content0.flag && face_content1.flag && face_content2.flag)
  {
    float sim01 = lite::cv::utils::math::cosine_similarity<float>(
        face_content0.embedding, face_content1.embedding);
    float sim02 = lite::cv::utils::math::cosine_similarity<float>(
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

人脸识别更多可以使用的算法包括：
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

</details>


<details>
<summary> 3.3 语义分割应用案例 </summary>

#### 3.3 使用[DeepLabV3ResNet101](https://pytorch.org/hub/pytorch_vision_deeplabv3_resnet101/) 进行语义分割。请从Model-Zoo[<sup>2</sup>](#lite.ai.toolkit-2) 下载模型文件。
```c++
#include "lite/lite.h"

static void test_default()
{
  std::string onnx_path = "../../../hub/onnx/cv/deeplabv3_resnet101_coco.onnx";
  std::string test_img_path = "../../../examples/lite/resources/test_lite_deeplabv3_resnet101.png";
  std::string save_img_path = "../../../logs/test_lite_deeplabv3_resnet101.jpg";

  auto *deeplabv3_resnet101 = new lite::cv::segmentation::DeepLabV3ResNet101(onnx_path, 16); // 16 threads

  lite::cv::types::SegmentContent content;
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

语义分割更多可用的算法包括：
```c++
auto *segment = new lite::cv::segmentation::FCNResNet101(onnx_path);
```

</details>


<details>
<summary> 3.4 人脸属性估计应用案例 </summary>

#### 3.4 使用[SSRNet](https://github.com/oukohou/SSR_Net_Pytorch) 进行年龄估计。请从Model-Zoo[<sup>2</sup>](#lite.ai.toolkit-2) 下载模型文件。
```c++
#include "lite/lite.h"

static void test_default()
{
  std::string onnx_path = "../../../hub/onnx/cv/ssrnet.onnx";
  std::string test_img_path = "../../../examples/lite/resources/test_lite_ssrnet.jpg";
  std::string save_img_path = "../../../logs/test_lite_ssrnet.jpg";

  lite::cv::face::attr::SSRNet *ssrnet = new lite::cv::face::attr::SSRNet(onnx_path);

  lite::cv::types::Age age;
  cv::Mat img_bgr = cv::imread(test_img_path);
  ssrnet->detect(img_bgr, age);
  lite::cv::utils::draw_age_inplace(img_bgr, age);
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

人脸属性估计更多可用的算法包括：
```c++
auto *attribute = new lite::cv::face::attr::AgeGoogleNet(onnx_path);  
auto *attribute = new lite::cv::face::attr::GenderGoogleNet(onnx_path); 
auto *attribute = new lite::cv::face::attr::EmotionFerPlus(onnx_path);
auto *attribute = new lite::cv::face::attr::VGG16Age(onnx_path);
auto *attribute = new lite::cv::face::attr::VGG16Gender(onnx_path);
auto *attribute = new lite::cv::face::attr::EfficientEmotion7(onnx_path); // 7 emotions, 15Mb only!
auto *attribute = new lite::cv::face::attr::EfficientEmotion8(onnx_path); // 8 emotions, 15Mb only!
auto *attribute = new lite::cv::face::attr::MobileEmotion7(onnx_path); // 7 emotions
auto *attribute = new lite::cv::face::attr::ReXNetEmotion7(onnx_path); // 7 emotions
```

</details>



<details>
<summary> 3.5 图像分类应用案例 </summary>

#### 3.5 使用[DenseNet](https://pytorch.org/hub/pytorch_vision_densenet/) 进行图像分类。请从Model-Zoo[<sup>2</sup>](#lite.ai.toolkit-2) 下载模型文件。
```c++
#include "lite/lite.h"

static void test_default()
{
  std::string onnx_path = "../../../hub/onnx/cv/densenet121.onnx";
  std::string test_img_path = "../../../examples/lite/resources/test_lite_densenet.jpg";

  auto *densenet = new lite::cv::classification::DenseNet(onnx_path);

  lite::cv::types::ImageNetContent content;
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

图像分类更多可用的算法包括：
```c++
auto *classifier = new lite::cv::classification::EfficientNetLite4(onnx_path);  
auto *classifier = new lite::cv::classification::ShuffleNetV2(onnx_path); 
auto *classifier = new lite::cv::classification::GhostNet(onnx_path);
auto *classifier = new lite::cv::classification::HdrDNet(onnx_path);
auto *classifier = new lite::cv::classification::IBNNet(onnx_path);
auto *classifier = new lite::cv::classification::MobileNetV2(onnx_path); 
auto *classifier = new lite::cv::classification::ResNet(onnx_path); 
auto *classifier = new lite::cv::classification::ResNeXt(onnx_path);
```

</details>



<details>
<summary> 3.6 人脸检测应用案例 </summary>

#### 3.6 使用[UltraFace](https://github.com/Linzaer/Ultra-Light-Fast-Generic-Face-Detector-1MB) 进行人脸检测。请从Model-Zoo[<sup>2</sup>](#lite.ai.toolkit-2) 下载模型文件。
```c++
#include "lite/lite.h"

static void test_default()
{
  std::string onnx_path = "../../../hub/onnx/cv/ultraface-rfb-640.onnx";
  std::string test_img_path = "../../../examples/lite/resources/test_lite_ultraface.jpg";
  std::string save_img_path = "../../../logs/test_lite_ultraface.jpg";

  auto *ultraface = new lite::cv::face::detect::UltraFace(onnx_path);

  std::vector<lite::cv::types::Boxf> detected_boxes;
  cv::Mat img_bgr = cv::imread(test_img_path);
  ultraface->detect(img_bgr, detected_boxes);
  lite::cv::utils::draw_boxes_inplace(img_bgr, detected_boxes);
  cv::imwrite(save_img_path, img_bgr);

  delete ultraface;
}
```
输出的结果是:
<div align='center'>
  <img src='logs/test_lite_ultraface.jpg' height="224px" width="224px">
  <img src='logs/test_lite_ultraface_2.jpg' height="224px" width="224px">
  <img src='logs/test_lite_ultraface_3.jpg' height="224px" width="224px">
</div>  

人脸检测更多可用的算法包括：
```c++
auto *detector = new lite::face::detect::UltraFace(onnx_path);  // 1.1Mb only !
auto *detector = new lite::face::detect::FaceBoxes(onnx_path);  // 3.8Mb only ! 
auto *detector = new lite::face::detect::RetinaFace(onnx_path);  // 1.6Mb only ! CVPR2020
```

</details>


<details>
<summary> 3.7 图像着色应用案例 </summary>

#### 3.7 使用[colorization](https://github.com/richzhang/colorization) 进行图像着色。请从Model-Zoo[<sup>2</sup>](#lite.ai.toolkit-2) 下载模型文件。
```c++
#include "lite/lite.h"

static void test_default()
{
  std::string onnx_path = "../../../hub/onnx/cv/eccv16-colorizer.onnx";
  std::string test_img_path = "../../../examples/lite/resources/test_lite_colorizer_1.jpg";
  std::string save_img_path = "../../../logs/test_lite_eccv16_colorizer_1.jpg";
  
  auto *colorizer = new lite::cv::colorization::Colorizer(onnx_path);
  
  cv::Mat img_bgr = cv::imread(test_img_path);
  lite::cv::types::ColorizeContent colorize_content;
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

</details>


<details>
<summary> 3.8 头部姿态估计应用案例 </summary>

#### 3.8 使用[FSANet](https://github.com/omasaht/headpose-fsanet-pytorch) 进行头部姿态估计。请从Model-Zoo[<sup>2</sup>](#lite.ai.toolkit-2) 下载模型文件。

```c++
#include "lite/lite.h"

static void test_default()
{
  std::string onnx_path = "../../../hub/onnx/cv/fsanet-var.onnx";
  std::string test_img_path = "../../../examples/lite/resources/test_lite_fsanet.jpg";
  std::string save_img_path = "../../../logs/test_lite_fsanet.jpg";

  auto *fsanet = new lite::cv::face::pose::FSANet(onnx_path);
  cv::Mat img_bgr = cv::imread(test_img_path);
  lite::cv::types::EulerAngles euler_angles;
  fsanet->detect(img_bgr, euler_angles);
  
  if (euler_angles.flag)
  {
    lite::cv::utils::draw_axis_inplace(img_bgr, euler_angles);
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

</details>


<details>
<summary> 3.9 人脸关键点检测应用案例 </summary>

#### 3.9 使用[FaceLandmarks1000](https://github.com/Single430/FaceLandmark1000) 进行人脸1000关键点检测。请从Model-Zoo[<sup>2</sup>](#lite.ai.toolkit-2) 下载模型文件。
```c++
#include "lite/lite.h"

static void test_default()
{
  std::string onnx_path = "../../../hub/onnx/cv/FaceLandmark1000.onnx";
  std::string test_img_path = "../../../examples/lite/resources/test_lite_face_landmarks_0.png";
  std::string save_img_path = "../../../logs/test_lite_face_landmarks_1000.jpg";
    
  auto *face_landmarks_1000 = new lite::cv::face::align::FaceLandmark1000(onnx_path);

  lite::cv::types::Landmarks landmarks;
  cv::Mat img_bgr = cv::imread(test_img_path);
  face_landmarks_1000->detect(img_bgr, landmarks);
  lite::cv::utils::draw_landmarks_inplace(img_bgr, landmarks);
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

人脸关键点检测更多可用的算法包括：
```c++
auto *align = new lite::cv::face::align::PFLD(onnx_path);  // 106 landmarks
auto *align = new lite::cv::face::align::PFLD98(onnx_path);  // 98 landmarks
auto *align = new lite::cv::face::align::PFLD68(onnx_path);  // 68 landmarks
auto *align = new lite::cv::face::align::MobileNetV268(onnx_path);  // 68 landmarks
auto *align = new lite::cv::face::align::MobileNetV2SE68(onnx_path);  // 68 landmarks
auto *align = new lite::cv::face::align::FaceLandmark1000(onnx_path);  // 1000 landmarks !
```

</details>


<details>
<summary> 3.10 风格迁移应用案例 </summary>

#### 3.10 使用[FastStyleTransfer](https://github.com/onnx/models/tree/master/vision/style_transfer/fast_neural_style) 进行自然风格迁移。请从Model-Zoo[<sup>2</sup>](#lite.ai.toolkit-2) 下载模型文件。
```c++
#include "lite/lite.h"

static void test_default()
{
  std::string onnx_path = "../../../hub/onnx/cv/style-candy-8.onnx";
  std::string test_img_path = "../../../examples/lite/resources/test_lite_fast_style_transfer.jpg";
  std::string save_img_path = "../../../logs/test_lite_fast_style_transfer_candy.jpg";
  
  auto *fast_style_transfer = new lite::cv::style::FastStyleTransfer(onnx_path);
 
  lite::cv::types::StyleContent style_content;
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

</details>


<details>
<summary> 3.11 抠图应用案例 </summary>

#### 3.11 使用[RobustVideoMatting](https://github.com/PeterL1n/RobustVideoMatting) 进行视频抠图。请从Model-Zoo[<sup>2</sup>](#lite.ai.toolkit-2) 下载模型文件。

```c++
#include "lite/lite.h"

static void test_default()
{
  std::string onnx_path = "../../../hub/onnx/cv/rvm_mobilenetv3_fp32.onnx";
  std::string video_path = "../../../examples/lite/resources/test_lite_rvm_0.mp4";
  std::string output_path = "../../../logs/test_lite_rvm_0.mp4";
  
  auto *rvm = new lite::cv::matting::RobustVideoMatting(onnx_path, 16); // 16 threads
  std::vector<lite::cv::types::MattingContent> contents;
  
  // 1. video matting.
  rvm->detect_video(video_path, output_path, contents);
  
  delete rvm;
}
```
输出的结果是:

<div align='center'>
  <img src='docs/resources/interviewi.gif' height="200px" width="200px">
  <img src='docs/resources/interview.gif' height="200px" width="200px">  
  <img src='docs/resources/dance3i.gif' height="200px" width="200px">
  <img src='docs/resources/dance3.gif' height="200px" width="200px">
  <br>
  <img src='docs/resources/teslai.gif' height="200px" width="200px">
  <img src='docs/resources/tesla.gif' height="200px" width="200px">  
  <img src='docs/resources/b5i.gif' height="200px" width="200px">
  <img src='docs/resources/b5.gif' height="200px" width="200px">
</div>


</details>

</details>


## 4. Lite.AI.ToolKit API文档

<div id="lite.ai.toolkit-Lite.AI.ToolKit-API-Docs"></div>

### 4.1 默认版本的APIs.
更多默认版本的API文档，详见 [api.default.md](https://github.com/DefTruth/lite.ai.toolkit/blob/main/docs/api/api.default.md) 。 比如，YoloV5的API是:

> `lite::cv::detection::YoloV5`
```c++
void detect(const cv::Mat &mat, std::vector<types::Boxf> &detected_boxes, 
            float score_threshold = 0.25f, float iou_threshold = 0.45f,
            unsigned int topk = 100, unsigned int nms_type = NMS::OFFSET);
```


<details>
<summary> ONNXRuntime，MNN 和 NCNN 版本的APIs.</summary>

### 4.2 ONNXRuntime 版本 APIs.
更多ONNXRuntime版本的API文档，详见 [api.onnxruntime.md](https://github.com/DefTruth/lite.ai.toolkit/blob/main/docs/api/api.onnxruntime.md) 。比如，YoloV5的API是:

> `lite::onnxruntime::cv::detection::YoloV5`
```c++
void detect(const cv::Mat &mat, std::vector<types::Boxf> &detected_boxes, 
            float score_threshold = 0.25f, float iou_threshold = 0.45f,
            unsigned int topk = 100, unsigned int nms_type = NMS::OFFSET);
```


### 4.3 MNN 版本 APIs.

(*todo*⚠️: 待实现)

> `lite::mnn::cv::detection::YoloV5`

> `lite::mnn::cv::detection::YoloV4`

> `lite::mnn::cv::detection::YoloV3`

> `lite::mnn::cv::detection::SSD`

...


### 4.4 NCNN 版本 APIs.

(*todo*⚠️: 待实现)

> `lite::ncnn::cv::detection::YoloV5`

> `lite::ncnn::cv::detection::YoloV4`

> `lite::ncnn::cv::detection::YoloV3`

> `lite::ncnn::cv::detection::SSD`

...

</details>



## 5. 其他文档

<div id="lite.ai.toolkit-Other-Docs"></div>  
<div id="lite.ai.toolkit-1"></div>

<details>
<summary> 展开其他文档 </summary>

### 5.1 ONNXRuntime相关的文档.
* [Rapid implementation of your inference using BasicOrtHandler](https://github.com/DefTruth/lite.ai.toolkit/blob/main/docs/ort/ort_handler.zh.md)
* [Some very useful onnxruntime c++ interfaces](https://github.com/DefTruth/lite.ai.toolkit/blob/main/docs/ort/ort_useful_api.zh.md)
* [How to compile a single model in this library you needed](https://github.com/DefTruth/lite.ai.toolkit/blob/main/docs/ort/ort_build_single.zh.md)
* [How to convert SubPixelCNN to ONNX and implements with onnxruntime c++](https://github.com/DefTruth/lite.ai.toolkit/blob/main/docs/ort/ort_subpixel_cnn.zh.md)
* [How to convert Colorizer to ONNX and implements with onnxruntime c++](https://github.com/DefTruth/lite.ai.toolkit/blob/main/docs/ort/ort_colorizer.zh.md)
* [How to convert SSRNet to ONNX and implements with onnxruntime c++](https://github.com/DefTruth/lite.ai.toolkit/blob/main/docs/ort/ort_ssrnet.zh.md)
* [How to convert YoloV3 to ONNX and implements with onnxruntime c++](https://github.com/DefTruth/lite.ai.toolkit/blob/main/docs/ort/ort_yolov3.zh.md)
* [How to convert YoloV5 to ONNX and implements with onnxruntime c++](https://github.com/DefTruth/lite.ai.toolkit/blob/main/docs/ort/ort_yolov5.zh.md)


### 5.2 [third_party](https://github.com/DefTruth/lite.ai.toolkit/tree/main/third_party) 相关的文档  

|Library|Target|Docs|
|:---:|:---:|:---:|
|OpenCV| mac-x86_64 | [opencv-mac-x86_64-build.zh.md](https://github.com/DefTruth/lite.ai.toolkit/blob/main/docs/third_party/opencv-mac-x86_64-build.zh.md) |
|OpenCV| android-arm | [opencv-static-android-arm-build.zh.md](https://github.com/DefTruth/lite.ai.toolkit/blob/main/docs/third_party/opencv-static-android-arm-build.zh.md) |
|onnxruntime| mac-x86_64 | [onnxruntime-mac-x86_64-build.zh.md](https://github.com/DefTruth/lite.ai.toolkit/blob/main/docs/third_party/onnxruntime-mac-x86_64-build.zh.md) |
|onnxruntime| android-arm | [onnxruntime-android-arm-build.zh.md](https://github.com/DefTruth/lite.ai.toolkit/blob/main/docs/third_party/onnxruntime-android-arm-build.zh.md) |
|NCNN| mac-x86_64 | todo⚠️ |
|MNN| mac-x86_64 | todo⚠️ |
|TNN| mac-x86_64 | todo⚠️ |

</details>

## 6. 开源协议 

<div id="lite.ai.toolkit-License"></div>

[Lite.AI.ToolKit](#lite.ai.toolkit-Introduction) 的代码采用GPL-3.0协议。


## 7. 引用参考 

<div id="lite.ai.toolkit-References"></div>

本项目参考了以下开源项目。

* [RobustVideoMatting](https://github.com/PeterL1n/RobustVideoMatting) (🔥🔥🔥new!!↑)
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

## 引用本项目 ![](https://img.shields.io/github/stars/DefTruth/lite.ai.toolkit.svg?style=social) ![](https://img.shields.io/github/forks/DefTruth/lite.ai.toolkit.svg?style=social) ![](https://img.shields.io/github/watchers/DefTruth/lite.ai.toolkit.svg?style=social)


如果您在自己的项目中使用了*Lite.AI.ToolKit*，可考虑按以下方式进行引用。整理不易，欢迎关注，🌟点赞收藏~ 🙃🤪🍀
```BibTeX
@misc{lite.ai.toolkit2021,
  title={lite.ai.toolkit: A lite C++ toolkit of awesome AI models.},
  url={https://github.com/DefTruth/lite.ai.toolkit},
  note={Open-source software available at https://github.com/DefTruth/lite.ai.toolkit},
  author={Yan Jun},
  year={2021}
}
```  



