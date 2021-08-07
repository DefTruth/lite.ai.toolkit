

# Lite.AI 🚀🚀🌟  

<div align='center'>
  <img src='logs/test_lite_yolov5_1.jpg' height="200px" width="200px">
  <img src='logs/test_lite_deeplabv3_resnet101.jpg' height="200px" width="200px">
  <img src='logs/test_lite_ssd_mobilenetv1.jpg' height="200px" width="200px">
  <img src='logs/test_lite_ultraface.jpg' height="200px" width="200px">
  <br> 
  <img src='logs/test_lite_face_landmarks_1000.jpg' height="200px" width="200px">
  <img src='logs/test_lite_fsanet.jpg' height="200px" width="200px">
  <img src='logs/test_lite_fast_style_transfer_candy.jpg' height="200px" width="200px">
  <img src='logs/test_lite_fast_style_transfer_mosaic.jpg' height="200px" width="200px"> 
</div>   


## Introduction.    

[![](https://img.shields.io/badge/MacOS-success-green.svg)](https://github.com/DefTruth/lite.ai/releases/tag/v0.0.1) ![](https://img.shields.io/badge/Linux-doing-yellowgreen.svg) [![](https://img.shields.io/badge/Version-0.0.1-brightgreen.svg)](https://github.com/DefTruth/lite.ai/releases/tag/v0.0.1) ![](https://img.shields.io/badge/Language-C%2B%2B-orange.svg) ![](https://img.shields.io/badge/License-MIT-blue.svg)


<div id="lite.ai-Introduction"></div> 

*Lite.AI* 🚀🚀🌟 is a simple and user-friendly C++ library of awesome🔥🔥🔥 AI models. It's a collection of personal interests. such as YOLOX, YoloV5, YoloV4, DeepLabV3, ArcFace, etc. *Lite.AI* based on *[onnxruntime c++](https://github.com/microsoft/onnxruntime)* by default. I do have plans to reimplement it with *[ncnn](https://github.com/Tencent/ncnn)* and *[MNN](https://github.com/alibaba/MNN)*, but not coming soon. It includes [object detection](#lite.ai-object-detection), [face detection](#lite.ai-face-detection), [style transfer](#lite.ai-style-transfer), [face alignment](#lite.ai-face-alignment), [face recognition](#lite.ai-face-recognition), [segmentation](#lite.ai-segmentation), [colorization](#lite.ai-colorization), [face attributes analysis](#lite.ai-face-attributes-analysis), [image classification](#lite.ai-image-classification), [matting](#lite.ai-matting), etc. You can use these awesome models simply through *lite::cv::Type::Class* syntax, such as *[lite::cv::detection::YoloV5](#lite.ai-object-detection)*. Star 🌟👆🏻 this repo if it does any helps to you ~ Have a good travel ~ 🙃🤪🍀

## Important Notes !!!  

* 🔥 (20210807) Added [YoloR](https://github.com/WongKinYiu/yolor) to *Lite.AI* ! Use it through [*lite::cv::detection::YoloR*](#lite.ai-object-detection) syntax ! See [demo](https://github.com/DefTruth/lite.ai/blob/main/examples/lite/cv/test_lite_yolor.cpp).
* ⚠️ (20210802) Added GPU Compatibility for CUDAExecutionProvider. See [issue#10](https://github.com/DefTruth/lite.ai/issues/10).
* ⚠️ (20210801) fixed [issue#9](https://github.com/DefTruth/lite.ai/issues/9) YOLOX inference error for non-square shape. See [yolox.cpp](https://github.com/DefTruth/lite.ai/blob/main/ort/cv/yolox.cpp).
* ✅ (20210731) Added [RetinaFace-CVPR2020](https://github.com/biubug6/Pytorch_Retinaface) for face detection, 1.6Mb only! See [demo](https://github.com/DefTruth/lite.ai/blob/main/examples/lite/cv/test_lite_retinaface.cpp).
* 🔥 (20210728) Added [FaceLandmarks1000](https://github.com/Single430/FaceLandmark1000) for 1000 facial landmarks detection, 2Mb only! See [demo](https://github.com/DefTruth/lite.ai/blob/main/examples/lite/cv/test_lite_face_landmarks_1000.cpp).
* ✅ (20210722) Update [lite.ai.hub.onnx.md](https://github.com/DefTruth/lite.ai/tree/main/docs/hub/lite.ai.hub.onnx.md) ! *Lite.AI* contains *60+* AI models with *100+* .onnx files now.
* 🔥 (20210721) Added [YOLOX](https://github.com/Megvii-BaseDetection/YOLOX) to *Lite.AI* ! Use it through [*lite::cv::detection::YoloX*](#lite.ai-object-detection) syntax ! See [demo](https://github.com/DefTruth/lite.ai/blob/main/examples/lite/cv/test_lite_yolox.cpp).  


<details>
<summary> Expand for More Notes.</summary>  

## More Notes !!!    

* ✅ (20210807) Added [TinyYoloV4VOC](https://github.com/bubbliiiing/yolov4-tiny-pytorch) for object detection! See [demo](https://github.com/DefTruth/lite.ai/blob/main/examples/lite/cv/test_lite_tiny_yolov4_voc.cpp).  
* ✅ (20210807) Added [TinyYoloV4COCO](https://github.com/bubbliiiing/yolov4-tiny-pytorch) for object detection! See [demo](https://github.com/DefTruth/lite.ai/blob/main/examples/lite/cv/test_lite_tiny_yolov4_coco.cpp).  
* ✅ (20210801) Added [FaceBoxes](https://github.com/zisianw/FaceBoxes.PyTorch) for face detection! See [demo](https://github.com/DefTruth/lite.ai/blob/main/examples/lite/cv/test_lite_faceboxes.cpp).  
* ✅ (20210727) Added [MobileNetV2SE68、PFLD68](https://github.com/cunjian/pytorch_face_landmark) for 68 facial landmarks detection! See [demo](https://github.com/DefTruth/lite.ai/blob/main/examples/lite/cv/test_lite_pfld68.cpp).  
* ✅ (20210726) Added [PFLD98](https://github.com/polarisZhao/PFLD-pytorch) for 98 facial landmarks detection! See [demo](https://github.com/DefTruth/lite.ai/blob/main/examples/lite/cv/test_lite_pfld98.cpp).  
* ⚠️ (20210716) *Lite.AI* was rename from the *LiteHub* repo! *LiteHub* will no longer be maintained.    


## Working Notes. 👇🏻  
  
  * ✅ [object detection](#lite.ai-object-detection) 
  * ✅ [image classification](#lite.ai-object-detection) 
  * ❇️ [face detection](#lite.ai-face-detection) 
  * ✅ [face alignment](#lite.ai-face-alignment) 
  * ✅ [face recognition](#lite.ai-face-recognition) 
  * ✅ [face attributes analysis](#lite.ai-face-attributes-analysis)
  * ⚠️ [segmentation](#lite.ai-segmentation)
  * ⚠️ [style transfer](#lite.ai-style-transfer)
  * ⚠️ [colorization](#lite.ai-colorization)
  * ⚠️ [matting](#lite.ai-matting)
  
</details>

## Contents.
* [Dependencies](#lite.ai-Dependencies)
* [Build Lite.AI](#lite.ai-Build-Lite.AI)
* [Model Zoo](#lite.ai-Model-Zoo)
* [Examples for LiteHub](#lite.ai-Examples-for-Lite.AI)

<!---- 
* [Contributions](#lite.ai-Contributions)
---->

## 1. Dependencies.  

<div id="lite.ai-Dependencies"></div>

### Mac OS.  

install `OpenCV` and `onnxruntime` libraries using Homebrew or you can download the built dependencies from this repo. See [third_party](https://github.com/DefTruth/litehub/tree/main/third_party) and build-docs[<sup>1</sup>](#lite.ai-1) for more details.

```shell
  brew update
  brew install opencv
  brew install onnxruntime
```  

<details>
<summary> Expand for More Details of Dependencies.</summary>  

### Linux.  
  * *todo*⚠️  
  
### Windows.   
  * *todo*⚠️  
  
### Inference Engine Plans:
  * *doing*:  
    ❇️ `onnxruntime` 
  * *todo*:  
    ⚠️ `NCNN`  
    ⚠️ `MNN`  
    ⚠️ `OpenMP`

</details>  


## 2. Build Lite.AI.

<div id="lite.ai-Build-Lite.AI"></div>

Build the shared lib of Lite.AI for *MacOS* from sources. Note that Lite.AI uses `onnxruntime` as default backend, for the reason that onnxruntime supports the most of onnx's operators. 


<details>
<summary> Linux and Windows. </summary>  

### Linux and Windows.  

⚠️ Lite.AI is not directly support Linux and Windows now. For Linux and Windows, you need to build or download(if have official builts) the shared libs of *OpenCV* and *ONNXRuntime* firstly and put then into the *third_party* directory. Please reference the build-docs[<sup>1</sup>](#lite.ai-1) for *third_party*.   


* Windows: You can reference to [issue#6](https://github.com/DefTruth/lite.ai/issues/6)  
* Linux: The Docs and Docker image for Linux will be coming soon ~ [issue#2](https://github.com/DefTruth/lite.ai/issues/2)  
* Happy News !!! : 🚀 You can download the latest *ONNXRuntime* official built libs of Windows, Linux, MacOS and Arm !!! Both CPU and GPU versions are available. No more attentions needed pay to build it from source. Download the official built libs from [v1.8.1](https://github.com/microsoft/onnxruntime/releases). I have used version 1.7.0 for Lite.AI now, you can downlod it from [v1.7.0](https://github.com/microsoft/onnxruntime/releases/tag/v1.7.0), but version 1.8.1 should also work, I guess ~  🙃🤪🍀. For *OpenCV*, try to build from source(Linux) or down load the official built(Windows) from [OpenCV 4.5.3](https://github.com/opencv/opencv/releases). Then put the includes and libs into *third_party* directory of Lite.AI. 

</details>  



* Clone the Lite.AI from sources:
```shell
git clone --depth=1 https://github.com/DefTruth/lite.ai.git  # latest
```  

<!---
* For users in China, you can try:
```shell
git clone --depth=1 https://github.com.cnpmjs.org/DefTruth/lite.ai.git  # latest
```  
--->

* Build shared lib.
```shell
cd lite.ai
sh ./build.sh
```
```shell
cd ./build/lite.ai/lib && otool -L liblite.ai.0.0.1.dylib 
liblite.ai.0.0.1.dylib:
        @rpath/liblite.ai.0.0.1.dylib (compatibility version 0.0.1, current version 0.0.1)
        @rpath/libopencv_highgui.4.5.dylib (compatibility version 4.5.0, current version 4.5.2)
        @rpath/libonnxruntime.1.7.0.dylib (compatibility version 0.0.0, current version 1.7.0)
        ...
```

* GPU Compatibility: See [issue#10](https://github.com/DefTruth/lite.ai/issues/10).

<details>
<summary> Expand for more details of How to link the shared lib of Lite.AI?</summary>

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
    └── liblite.ai.0.0.1.dylib
```
* Run the built examples:
```shell
cd ./build/lite.ai/bin && ls -lh | grep lite
-rwxr-xr-x  1 root  staff   301K Jun 26 23:10 liblite.ai.0.0.1.dylib
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

* To link `lite.ai` shared lib. You need to make sure that `OpenCV` and `onnxruntime` are linked correctly. Just like:

```cmake
cmake_minimum_required(VERSION 3.17)
project(testlite.ai)
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
# link lite.ai.
set(LITEHUB_DIR ${CMAKE_SOURCE_DIR}/lite.ai)
set(LITEHUB_INCLUDE_DIR ${LITEHUB_DIR}/include)
set(LITEHUB_LIBRARY_DIR ${LITEHUB_DIR}/lib)
include_directories(${LITEHUB_INCLUDE_DIR})
link_directories(${LITEHUB_LIBRARY_DIR})
# add your executable
add_executable(lite_yolov5 test_lite_yolov5.cpp)
target_link_libraries(lite_yolov5 lite.ai onnxruntime ${OpenCV_LIBS})
```
A minimum example to show you how to link the shared lib of Lite.AI correctly for your own project can be found at [lite.ai-release](https://github.com/DefTruth/lite.ai-release) .

</details>


## 3. Model Zoo.

<div id="lite.ai-Model-Zoo"></div>

### 3.1 Namespace and Lite.AI modules.   
*Lite.AI* contains *60+* AI models with *100+* frozen pretrained *.onnx* files now. They come from different fields of computer vision. Click the Expand ▶️ button for more details.

<details>
<summary> Expand Details for Namespace and Lite.AI modules.</summary>  

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

</details>

### 3.2 Lite.AI's Classes and Pretrained Files.  

Correspondence between the classes in *Lite.AI* and pretrained model files can be found at [lite.ai.hub.onnx.md](https://github.com/DefTruth/lite.ai/tree/main/docs/hub/lite.ai.hub.onnx.md). For examples, the pretrained model files for *lite::cv::detection::YoloV5* and *lite::cv::detection::YoloX* are listed as follows.

<details>
<summary> Expand Examples for Lite.AI's Classes and Pretrained Files.</summary>  


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

It means that you can load the the any one `yolov5*.onnx` and  `yolox_*.onnx` according to your application through the same Lite.AI classes, such as *YoloV5*, *YoloX*, etc.

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


### 3.3 Model Zoo for Lite.AI.

Note that the models here are all from third-party projects. Most of the models were converted by *Lite.AI*. In Lite.AI, different names of the same algorithm mean that the corresponding models come from different repositories, different implementations, or use different training data, etc. ✅ means passed the test and ⚠️ means not implements yet but coming soon. For classes which denoted ✅, you can use it through *lite::cv::Type::Class* syntax, such as *[lite::cv::detection::YoloV5](#lite.ai-object-detection)* . More details can be found at [Examples for Lite.AI](#lite.ai-Examples-for-Lite.AI) .  
([Baidu Drive](https://pan.baidu.com/s/1elUGcx7CZkkjEoYhTMwTRQ) code: 8gin) <div id="lite.ai-2"></div>

<!---
For example, ArcFace in [insightface](https://github.com/deepinsight/insightface) is different from ArcFace in [face.evoLVe.PyTorch](https://github.com/ZhaoJ9014/face.evoLVe.PyTorch) . ArcFace in [insightface](https://github.com/deepinsight/insightface) uses Arc-Loss + Softmax, while ArcFace in [face.evoLVe.PyTorch](https://github.com/ZhaoJ9014/face.evoLVe.PyTorch) uses Arc-Loss + Focal-Loss. Lite.AI uses naming to make the necessary distinctions between models from different sources.
---->

* Object Detection.  

|Class|Size|From|Awesome|File|Type|State|Usage|
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
|[YoloV5](https://github.com/ultralytics/yolov5)|28M|[yolov5](https://github.com/ultralytics/yolov5)|🔥🔥💥↑| [![](https://img.shields.io/badge/onnx-done-brightgreen.svg)](https://github.com/DefTruth/lite.ai/tree/main/docs/hub/lite.ai.hub.onnx.md#lite.ai.hub.onnx-object-detection)  | *detection* | ✅ | [demo](https://github.com/DefTruth/lite.ai/blob/main/examples/lite/cv/test_lite_yolov5.cpp) |
|[YoloV3](https://github.com/onnx/models/blob/master/vision/object_detection_segmentation/yolov3)|236M|[onnx-models](https://github.com/onnx/models)|🔥🔥🔥↑| [![](https://img.shields.io/badge/onnx-done-brightgreen.svg)](https://github.com/DefTruth/lite.ai/tree/main/docs/hub/lite.ai.hub.onnx.md#lite.ai.hub.onnx-object-detection) | *detection* | ✅ | [demo](https://github.com/DefTruth/lite.ai/blob/main/examples/lite/cv/test_lite_yolov3.cpp) |
|[TinyYoloV3](https://github.com/onnx/models/blob/master/vision/object_detection_segmentation/tiny-yolov3)|33M|        [onnx-models](https://github.com/onnx/models)         | 🔥🔥🔥↑ | [![](https://img.shields.io/badge/onnx-done-brightgreen.svg)](https://github.com/DefTruth/lite.ai/tree/main/docs/hub/lite.ai.hub.onnx.md#lite.ai.hub.onnx-object-detection) | *detection* | ✅ | [demo](https://github.com/DefTruth/lite.ai/blob/main/examples/lite/cv/test_lite_tiny_yolov3.cpp) |
|[YoloV4](https://github.com/argusswift/YOLOv4-pytorch)|176M| [YOLOv4...](https://github.com/argusswift/YOLOv4-pytorch) | 🔥🔥🔥↑ | [![](https://img.shields.io/badge/onnx-done-brightgreen.svg)](https://github.com/DefTruth/lite.ai/tree/main/docs/hub/lite.ai.hub.onnx.md#lite.ai.hub.onnx-object-detection) | *detection* | ✅ | [demo](https://github.com/DefTruth/lite.ai/blob/main/examples/lite/cv/test_lite_yolov4.cpp) |
|[SSD](https://github.com/onnx/models/blob/master/vision/object_detection_segmentation/ssd)|76M|        [onnx-models](https://github.com/onnx/models)         | 🔥🔥🔥↑ | [![](https://img.shields.io/badge/onnx-done-brightgreen.svg)](https://github.com/DefTruth/lite.ai/tree/main/docs/hub/lite.ai.hub.onnx.md#lite.ai.hub.onnx-object-detection) | *detection* | ✅ | [demo](https://github.com/DefTruth/lite.ai/blob/main/examples/lite/cv/test_lite_ssd.cpp) |
|[SSDMobileNetV1](https://github.com/onnx/models/blob/master/vision/object_detection_segmentation/ssd-mobilenetv1)|27M|        [onnx-models](https://github.com/onnx/models)         | 🔥🔥🔥↑ | [![](https://img.shields.io/badge/onnx-done-brightgreen.svg)](https://github.com/DefTruth/lite.ai/tree/main/docs/hub/lite.ai.hub.onnx.md#lite.ai.hub.onnx-object-detection) | *detection* | ✅ | [demo](https://github.com/DefTruth/lite.ai/blob/main/examples/lite/cv/test_lite_ssd_mobilenetv1.cpp) |
|[YoloX](https://github.com/Megvii-BaseDetection/YOLOX)|3.5M| [YOLOX](https://github.com/Megvii-BaseDetection/YOLOX) | 🔥🔥new↑ | [![](https://img.shields.io/badge/onnx-done-brightgreen.svg)](https://github.com/DefTruth/lite.ai/tree/main/docs/hub/lite.ai.hub.onnx.md#lite.ai.hub.onnx-object-detection) | *detection* | ✅ | [demo](https://github.com/DefTruth/lite.ai/blob/main/examples/lite/cv/test_lite_yolox.cpp) |
|[TinyYoloV4VOC](https://github.com/bubbliiiing/yolov4-tiny-pytorch)|22M| [yolov4-tiny...](https://github.com/bubbliiiing/yolov4-tiny-pytorch) | 🔥🔥↑ | [![](https://img.shields.io/badge/onnx-done-brightgreen.svg)](https://github.com/DefTruth/lite.ai/tree/main/docs/hub/lite.ai.hub.onnx.md#lite.ai.hub.onnx-object-detection) | *detection* | ✅ | [demo](https://github.com/DefTruth/lite.ai/blob/main/examples/lite/cv/test_lite_tiny_yolov4_voc.cpp) |
|[TinyYoloV4COCO](https://github.com/bubbliiiing/yolov4-tiny-pytorch)|22M| [yolov4-tiny...](https://github.com/bubbliiiing/yolov4-tiny-pytorch) | 🔥🔥↑ | [![](https://img.shields.io/badge/onnx-done-brightgreen.svg)](https://github.com/DefTruth/lite.ai/tree/main/docs/hub/lite.ai.hub.onnx.md#lite.ai.hub.onnx-object-detection) | *detection* | ✅ | [demo](https://github.com/DefTruth/lite.ai/blob/main/examples/lite/cv/test_lite_tiny_yolov4_coco.cpp) |
|[YoloR](https://github.com/WongKinYiu/yolor)|39M| [yolor](https://github.com/WongKinYiu/yolor) | 🔥🔥new↑ | [![](https://img.shields.io/badge/onnx-done-brightgreen.svg)](https://github.com/DefTruth/lite.ai/tree/main/docs/hub/lite.ai.hub.onnx.md#lite.ai.hub.onnx-object-detection) | *detection* | ✅ | [demo](https://github.com/DefTruth/lite.ai/blob/main/examples/lite/cv/test_lite_yolor.cpp) |


* Face Recognition.  

|Class|Size|From|Awesome|File|Type|State|Usage|
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
|[GlintArcFace](https://github.com/deepinsight/insightface/tree/master/recognition/arcface_torch)|92M|  [insightface](https://github.com/deepinsight/insightface)   | 🔥🔥🔥↑ | [![](https://img.shields.io/badge/onnx-done-brightgreen.svg)](https://github.com/DefTruth/lite.ai/tree/main/docs/hub/lite.ai.hub.onnx.md#lite.ai.hub.onnx-face-recognition) | *faceid* | ✅ | [demo](https://github.com/DefTruth/lite.ai/blob/main/examples/lite/cv/test_lite_glint_arcface.cpp) |
|[GlintCosFace](https://github.com/deepinsight/insightface/tree/master/recognition/arcface_torch)|92M|  [insightface](https://github.com/deepinsight/insightface)   | 🔥🔥🔥↑ | [![](https://img.shields.io/badge/onnx-done-brightgreen.svg)](https://github.com/DefTruth/lite.ai/tree/main/docs/hub/lite.ai.hub.onnx.md#lite.ai.hub.onnx-face-recognition) | *faceid* | ✅ | [demo](https://github.com/DefTruth/lite.ai/blob/main/examples/lite/cv/test_lite_glint_cosface.cpp) |
|[GlintPartialFC](https://github.com/deepinsight/insightface/tree/master/recognition/partial_fc)|170M|  [insightface](https://github.com/deepinsight/insightface)   | 🔥🔥🔥↑ | [![](https://img.shields.io/badge/onnx-done-brightgreen.svg)](https://github.com/DefTruth/lite.ai/tree/main/docs/hub/lite.ai.hub.onnx.md#lite.ai.hub.onnx-face-recognition) | *faceid* | ✅ | [demo](https://github.com/DefTruth/lite.ai/blob/main/examples/lite/cv/test_lite_glint_partial_fc.cpp) |
|[FaceNet](https://github.com/timesler/facenet-pytorch)|89M| [facenet...](https://github.com/timesler/facenet-pytorch) | 🔥🔥🔥↑ | [![](https://img.shields.io/badge/onnx-done-brightgreen.svg)](https://github.com/DefTruth/lite.ai/tree/main/docs/hub/lite.ai.hub.onnx.md#lite.ai.hub.onnx-face-recognition) | *faceid* | ✅ | [demo](https://github.com/DefTruth/lite.ai/blob/main/examples/lite/cv/test_lite_facenet.cpp) |
|[FocalArcFace](https://github.com/ZhaoJ9014/face.evoLVe.PyTorch)|166M| [face.evoLVe...](https://github.com/ZhaoJ9014/face.evoLVe.PyTorch) | 🔥🔥🔥↑ | [![](https://img.shields.io/badge/onnx-done-brightgreen.svg)](https://github.com/DefTruth/lite.ai/tree/main/docs/hub/lite.ai.hub.onnx.md#lite.ai.hub.onnx-face-recognition) | *faceid* | ✅ | [demo](https://github.com/DefTruth/lite.ai/blob/main/examples/lite/cv/test_lite_focal_arcface.cpp) |
|[FocalAsiaArcFace](https://github.com/ZhaoJ9014/face.evoLVe.PyTorch)|166M| [face.evoLVe...](https://github.com/ZhaoJ9014/face.evoLVe.PyTorch) | 🔥🔥🔥↑ | [![](https://img.shields.io/badge/onnx-done-brightgreen.svg)](https://github.com/DefTruth/lite.ai/tree/main/docs/hub/lite.ai.hub.onnx.md#lite.ai.hub.onnx-face-recognition) | *faceid* | ✅ | [demo](https://github.com/DefTruth/lite.ai/blob/main/examples/lite/cv/test_lite_focal_asia_arcface.cpp) |
|[TencentCurricularFace](https://github.com/Tencent/TFace/tree/master/tasks/distfc)|249M|          [TFace](https://github.com/Tencent/TFace)           |  🔥🔥↑  | [![](https://img.shields.io/badge/onnx-done-brightgreen.svg)](https://github.com/DefTruth/lite.ai/tree/main/docs/hub/lite.ai.hub.onnx.md#lite.ai.hub.onnx-face-recognition) | *faceid* | ✅ | [demo](https://github.com/DefTruth/lite.ai/blob/main/examples/lite/cv/test_lite_tencent_curricular_face.cpp) |
|[TencentCifpFace](https://github.com/Tencent/TFace/tree/master/tasks/cifp)|130M|          [TFace](https://github.com/Tencent/TFace)           |  🔥🔥↑  | [![](https://img.shields.io/badge/onnx-done-brightgreen.svg)](https://github.com/DefTruth/lite.ai/tree/main/docs/hub/lite.ai.hub.onnx.md#lite.ai.hub.onnx-face-recognition) | *faceid* | ✅ | [demo](https://github.com/DefTruth/lite.ai/blob/main/examples/lite/cv/test_lite_tencent_cifp_face.cpp) |
|[CenterLossFace](https://github.com/louis-she/center-loss.pytorch)| 280M |  [center-loss...](https://github.com/louis-she/center-loss.pytorch)           |  🔥🔥↑  | [![](https://img.shields.io/badge/onnx-done-brightgreen.svg)](https://github.com/DefTruth/lite.ai/tree/main/docs/hub/lite.ai.hub.onnx.md#lite.ai.hub.onnx-face-recognition) | *faceid* | ✅ | [demo](https://github.com/DefTruth/lite.ai/blob/main/examples/lite/cv/test_lite_center_loss_face.cpp) |
|[SphereFace](https://github.com/clcarwin/sphereface_pytorch)| 80M |  [sphere...](https://github.com/clcarwin/sphereface_pytorch)   |  🔥🔥↑  | [![](https://img.shields.io/badge/onnx-done-brightgreen.svg)](https://github.com/DefTruth/lite.ai/tree/main/docs/hub/lite.ai.hub.onnx.md#lite.ai.hub.onnx-face-recognition) | *faceid* | ✅️ | [demo](https://github.com/DefTruth/lite.ai/blob/main/examples/lite/cv/test_lite_sphere_face.cpp) |
|[PoseRobustFace](https://github.com/penincillin/DREAM)| 92M | [DREAM](https://github.com/penincillin/DREAM)  |  🔥🔥↑  | [![](https://img.shields.io/badge/onnx-done-brightgreen.svg)](https://github.com/DefTruth/lite.ai/tree/main/docs/hub/lite.ai.hub.onnx.md#lite.ai.hub.onnx-face-recognition) | *faceid* | ✅️ | [demo](https://github.com/DefTruth/lite.ai/blob/main/examples/lite/cv/test_lite_pose_robust_face.cpp) |
|[NaivePoseRobustFace](https://github.com/penincillin/DREAM)| 43M | [DREAM](https://github.com/penincillin/DREAM)  |  🔥🔥↑  | [![](https://img.shields.io/badge/onnx-done-brightgreen.svg)](https://github.com/DefTruth/lite.ai/tree/main/docs/hub/lite.ai.hub.onnx.md#lite.ai.hub.onnx-face-recognition) | *faceid* | ✅️ | [demo](https://github.com/DefTruth/lite.ai/blob/main/examples/lite/cv/test_lite_naive_pose_robust_face.cpp) |
|[MobileFaceNet](https://github.com/Xiaoccer/MobileFaceNet_Pytorch)| 3.8M |  [MobileFace...](https://github.com/Xiaoccer/MobileFaceNet_Pytorch)           |  🔥🔥↑  | [![](https://img.shields.io/badge/onnx-done-brightgreen.svg)](https://github.com/DefTruth/lite.ai/tree/main/docs/hub/lite.ai.hub.onnx.md#lite.ai.hub.onnx-face-recognition) | *faceid* | ✅ | [demo](https://github.com/DefTruth/lite.ai/blob/main/examples/lite/cv/test_lite_mobile_facenet.cpp) |
|[CavaGhostArcFace](https://github.com/cavalleria/cavaface.pytorch)| 15M | [cavaface...](https://github.com/cavalleria/cavaface.pytorch) |  🔥🔥↑  | [![](https://img.shields.io/badge/onnx-done-brightgreen.svg)](https://github.com/DefTruth/lite.ai/tree/main/docs/hub/lite.ai.hub.onnx.md#lite.ai.hub.onnx-face-recognition) | *faceid* |  ✅ | [demo](https://github.com/DefTruth/lite.ai/blob/main/examples/lite/cv/test_lite_cava_ghost_arcface.cpp) |
|[CavaCombinedFace](https://github.com/cavalleria/cavaface.pytorch)| 250M | [cavaface...](https://github.com/cavalleria/cavaface.pytorch) |  🔥🔥↑  | [![](https://img.shields.io/badge/onnx-done-brightgreen.svg)](https://github.com/DefTruth/lite.ai/tree/main/docs/hub/lite.ai.hub.onnx.md#lite.ai.hub.onnx-face-recognition) | *faceid* | ✅ | [demo](https://github.com/DefTruth/lite.ai/blob/main/examples/lite/cv/test_lite_cava_combined_face.cpp) |
|[MobileSEFocalFace](https://github.com/grib0ed0v/face_recognition.pytorch)|4.5M| [face_recog...](https://github.com/grib0ed0v/face_recognition.pytorch) | 🔥🔥↑ | [![](https://img.shields.io/badge/onnx-done-brightgreen.svg)](https://github.com/DefTruth/lite.ai/tree/main/docs/hub/lite.ai.hub.onnx.md#lite.ai.hub.onnx-face-recognition) | *faceid* | ✅ |  [demo](https://github.com/DefTruth/lite.ai/blob/main/examples/lite/cv/test_lite_mobilese_focal_face.cpp) |

<details>
<summary> ⚠️ Expand More Details for Lite.AI's Model Zoo.</summary>  

* Face Detection.

|Class|Size|From|Awesome|File|Type|State|Usage|
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
|[UltraFace](https://github.com/Linzaer/Ultra-Light-Fast-Generic-Face-Detector-1MB)|1.1M| [Ultra-Light...](https://github.com/Linzaer/Ultra-Light-Fast-Generic-Face-Detector-1MB) | 🔥🔥🔥↑ | [![](https://img.shields.io/badge/onnx-done-brightgreen.svg)](https://github.com/DefTruth/lite.ai/tree/main/docs/hub/lite.ai.hub.onnx.md#lite.ai.hub.onnx-face-detection) | *face::detect* | ✅ | [demo](https://github.com/DefTruth/lite.ai/blob/main/examples/lite/cv/test_lite_ultraface.cpp) |
|[RetinaFace](https://github.com/biubug6/Pytorch_Retinaface)|1.6M| [...Retinaface](https://github.com/biubug6/Pytorch_Retinaface) | 🔥🔥🔥↑ | [![](https://img.shields.io/badge/onnx-done-brightgreen.svg)](https://github.com/DefTruth/lite.ai/tree/main/docs/hub/lite.ai.hub.onnx.md#lite.ai.hub.onnx-face-detection) | *face::detect* | ✅ | [demo](https://github.com/DefTruth/lite.ai/blob/main/examples/lite/cv/test_lite_retinaface.cpp) |
|[FaceBoxes](https://github.com/zisianw/FaceBoxes.PyTorch)|3.8M| [FaceBoxes](https://github.com/zisianw/FaceBoxes.PyTorch) | 🔥🔥↑ | [![](https://img.shields.io/badge/onnx-done-brightgreen.svg)](https://github.com/DefTruth/lite.ai/tree/main/docs/hub/lite.ai.hub.onnx.md#lite.ai.hub.onnx-face-detection) | *face::detect* | ✅ | [demo](https://github.com/DefTruth/lite.ai/blob/main/examples/lite/cv/test_lite_faceboxes.cpp) |


* Face Alignment.

|Class|Size|From|Awesome|File|Type|State|Usage|
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
|[PFLD](https://github.com/Hsintao/pfld_106_face_landmarks)|1.0M| [pfld_106_...](https://github.com/Hsintao/pfld_106_face_landmarks) |  🔥🔥↑  | [![](https://img.shields.io/badge/onnx-done-brightgreen.svg)](https://github.com/DefTruth/lite.ai/tree/main/docs/hub/lite.ai.hub.onnx.md#lite.ai.hub.onnx-face-alignment) | *face::align* | ✅ | [demo](https://github.com/DefTruth/lite.ai/blob/main/examples/lite/cv/test_lite_pfld.cpp) |
|[PFLD98](https://github.com/polarisZhao/PFLD-pytorch)|4.8M| [PFLD...](https://github.com/polarisZhao/PFLD-pytorch) | 🔥🔥↑ | [![](https://img.shields.io/badge/onnx-done-brightgreen.svg)](https://github.com/DefTruth/lite.ai/tree/main/docs/hub/lite.ai.hub.onnx.md#lite.ai.hub.onnx-face-alignment) | *face::align* | ✅️ | [demo](https://github.com/DefTruth/lite.ai/blob/main/examples/lite/cv/test_lite_pfld98.cpp) |
|[MobileNetV268](https://github.com/cunjian/pytorch_face_landmark)|9.4M| [...landmark](https://github.com/cunjian/pytorch_face_landmark) | 🔥🔥↑ |  [![](https://img.shields.io/badge/onnx-done-brightgreen.svg)](https://github.com/DefTruth/lite.ai/tree/main/docs/hub/lite.ai.hub.onnx.md#lite.ai.hub.onnx-face-alignment) | *face::align* | ✅️️ | [demo](https://github.com/DefTruth/lite.ai/blob/main/examples/lite/cv/test_lite_mobilenetv2_68.cpp) |
|[MobileNetV2SE68](https://github.com/cunjian/pytorch_face_landmark)|11M| [...landmark](https://github.com/cunjian/pytorch_face_landmark) | 🔥🔥↑ |  [![](https://img.shields.io/badge/onnx-done-brightgreen.svg)](https://github.com/DefTruth/lite.ai/tree/main/docs/hub/lite.ai.hub.onnx.md#lite.ai.hub.onnx-face-alignment) | *face::align* | ✅️️ | [demo](https://github.com/DefTruth/lite.ai/blob/main/examples/lite/cv/test_lite_mobilenetv2_se_68.cpp) |
|[PFLD68](https://github.com/cunjian/pytorch_face_landmark)|2.8M| [...landmark](https://github.com/cunjian/pytorch_face_landmark) | 🔥🔥↑ |  [![](https://img.shields.io/badge/onnx-done-brightgreen.svg)](https://github.com/DefTruth/lite.ai/tree/main/docs/hub/lite.ai.hub.onnx.md#lite.ai.hub.onnx-face-alignment) | *face::align* | ✅️ | [demo](https://github.com/DefTruth/lite.ai/blob/main/examples/lite/cv/test_lite_pfld68.cpp) |
|[FaceLandmark1000](https://github.com/Single430/FaceLandmark1000)|2.0M| [FaceLandm...](https://github.com/Single430/FaceLandmark1000) | 🔥↑ |  [![](https://img.shields.io/badge/onnx-done-brightgreen.svg)](https://github.com/DefTruth/lite.ai/tree/main/docs/hub/lite.ai.hub.onnx.md#lite.ai.hub.onnx-face-alignment) | *face::align* | ✅️ | [demo](https://github.com/DefTruth/lite.ai/blob/main/examples/lite/cv/test_lite_face_landmarks_1000.cpp) |



* Head Pose Estimation.  

|Class|Size|From|Awesome|File|Type|State|Usage|
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
|[FSANet](https://github.com/omasaht/headpose-fsanet-pytorch)|1.2M| [...fsanet...](https://github.com/omasaht/headpose-fsanet-pytorch) | 🔥↑ | [![](https://img.shields.io/badge/onnx-done-brightgreen.svg)](https://github.com/DefTruth/lite.ai/tree/main/docs/hub/lite.ai.hub.onnx.md#lite.ai.hub.onnx-head-pose-estimation) | *face::pose* | ✅ | [demo](https://github.com/DefTruth/lite.ai/blob/main/examples/lite/cv/test_lite_fsanet.cpp) |  


* Face Attributes.

|Class|Size|From|Awesome|File|Type|State|Usage|
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
|[AgeGoogleNet](https://github.com/onnx/models/tree/master/vision/body_analysis/age_gender)|23M|        [onnx-models](https://github.com/onnx/models)         | 🔥🔥🔥↑ | [![](https://img.shields.io/badge/onnx-done-brightgreen.svg)](https://github.com/DefTruth/lite.ai/tree/main/docs/hub/lite.ai.hub.onnx.md#lite.ai.hub.onnx-face-attributes) | *face::attr* | ✅ | [demo](https://github.com/DefTruth/lite.ai/blob/main/examples/lite/cv/test_lite_age_googlenet.cpp) |
|[GenderGoogleNet](https://github.com/onnx/models/tree/master/vision/body_analysis/age_gender)|23M|        [onnx-models](https://github.com/onnx/models)         | 🔥🔥🔥↑ | [![](https://img.shields.io/badge/onnx-done-brightgreen.svg)](https://github.com/DefTruth/lite.ai/tree/main/docs/hub/lite.ai.hub.onnx.md#lite.ai.hub.onnx-face-attributes) | *face::attr* | ✅ | [demo](https://github.com/DefTruth/lite.ai/blob/main/examples/lite/cv/test_lite_gender_googlenet.cpp) |
|[EmotionFerPlus](https://github.com/onnx/models/blob/master/vision/body_analysis/emotion_ferplus)|33M|        [onnx-models](https://github.com/onnx/models)         | 🔥🔥🔥↑ | [![](https://img.shields.io/badge/onnx-done-brightgreen.svg)](https://github.com/DefTruth/lite.ai/tree/main/docs/hub/lite.ai.hub.onnx.md#lite.ai.hub.onnx-face-attributes) | *face::attr* | ✅ | [demo](https://github.com/DefTruth/lite.ai/blob/main/examples/lite/cv/test_lite_emotion_ferplus.cpp) |
|[VGG16Age](https://github.com/onnx/models/tree/master/vision/body_analysis/age_gender)|514M|        [onnx-models](https://github.com/onnx/models)         | 🔥🔥🔥↑ | [![](https://img.shields.io/badge/onnx-done-brightgreen.svg)](https://github.com/DefTruth/lite.ai/tree/main/docs/hub/lite.ai.hub.onnx.md#lite.ai.hub.onnx-face-attributes) | *face::attr* | ✅ | [demo](https://github.com/DefTruth/lite.ai/blob/main/examples/lite/cv/test_lite_vgg16_age.cpp) |
|[VGG16Gender](https://github.com/onnx/models/tree/master/vision/body_analysis/age_gender)|512M|        [onnx-models](https://github.com/onnx/models)         | 🔥🔥🔥↑ | [![](https://img.shields.io/badge/onnx-done-brightgreen.svg)](https://github.com/DefTruth/lite.ai/tree/main/docs/hub/lite.ai.hub.onnx.md#lite.ai.hub.onnx-face-attributes) | *face::attr* | ✅ | [demo](https://github.com/DefTruth/lite.ai/blob/main/examples/lite/cv/test_lite_vgg16_gender.cpp) |
|[SSRNet](https://github.com/oukohou/SSR_Net_Pytorch)|190K| [SSR_Net...](https://github.com/oukohou/SSR_Net_Pytorch) | 🔥↑ | [![](https://img.shields.io/badge/onnx-done-brightgreen.svg)](https://github.com/DefTruth/lite.ai/tree/main/docs/hub/lite.ai.hub.onnx.md#lite.ai.hub.onnx-face-attributes) | *face::attr* | ✅ | [demo](https://github.com/DefTruth/lite.ai/blob/main/examples/lite/cv/test_lite_ssrnet.cpp) |
|[EfficientEmotion7](https://github.com/HSE-asavchenko/face-emotion-recognition)|15M| [face-emo...](https://github.com/HSE-asavchenko/face-emotion-recognition) | 🔥↑ | [![](https://img.shields.io/badge/onnx-done-brightgreen.svg)](https://github.com/DefTruth/lite.ai/tree/main/docs/hub/lite.ai.hub.onnx.md#lite.ai.hub.onnx-face-attributes) | *face::attr* | ✅️ | [demo](https://github.com/DefTruth/lite.ai/blob/main/examples/lite/cv/test_lite_efficient_emotion7.cpp) |
|[EfficientEmotion8](https://github.com/HSE-asavchenko/face-emotion-recognition)|15M| [face-emo...](https://github.com/HSE-asavchenko/face-emotion-recognition) | 🔥↑ | [![](https://img.shields.io/badge/onnx-done-brightgreen.svg)](https://github.com/DefTruth/lite.ai/tree/main/docs/hub/lite.ai.hub.onnx.md#lite.ai.hub.onnx-face-attributes) | *face::attr* | ✅  | [demo](https://github.com/DefTruth/lite.ai/blob/main/examples/lite/cv/test_lite_efficient_emotion8.cpp) |
|[MobileEmotion7](https://github.com/HSE-asavchenko/face-emotion-recognition)|13M| [face-emo...](https://github.com/HSE-asavchenko/face-emotion-recognition) | 🔥↑ | [![](https://img.shields.io/badge/onnx-done-brightgreen.svg)](https://github.com/DefTruth/lite.ai/tree/main/docs/hub/lite.ai.hub.onnx.md#lite.ai.hub.onnx-face-attributes) | *face::attr* |  ✅  | [demo](https://github.com/DefTruth/lite.ai/blob/main/examples/lite/cv/test_lite_mobile_emotion7.cpp) |
|[ReXNetEmotion7](https://github.com/HSE-asavchenko/face-emotion-recognition)|30M| [face-emo...](https://github.com/HSE-asavchenko/face-emotion-recognition) | 🔥↑ | [![](https://img.shields.io/badge/onnx-done-brightgreen.svg)](https://github.com/DefTruth/lite.ai/tree/main/docs/hub/lite.ai.hub.onnx.md#lite.ai.hub.onnx-face-attributes) | *face::attr* |  ✅  | [demo](https://github.com/DefTruth/lite.ai/blob/main/examples/lite/cv/test_lite_rexnet_emotion7.cpp) |


* Classification.

|Class|Size|From|Awesome|File|Type|State|Usage|
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
|[EfficientNetLite4](https://github.com/onnx/models/blob/master/vision/classification/efficientnet-lite4)|49M|        [onnx-models](https://github.com/onnx/models)         | 🔥🔥🔥↑ | [![](https://img.shields.io/badge/onnx-done-brightgreen.svg)](https://github.com/DefTruth/lite.ai/tree/main/docs/hub/lite.ai.hub.onnx.md#lite.ai.hub.onnx-classification)  | *classification* | ✅ | [demo](https://github.com/DefTruth/lite.ai/blob/main/examples/lite/cv/test_lite_efficientnet_lite4.cpp) |
|[ShuffleNetV2](https://github.com/onnx/models/blob/master/vision/classification/shufflenet)|8.7M|        [onnx-models](https://github.com/onnx/models)         | 🔥🔥🔥↑ | [![](https://img.shields.io/badge/onnx-done-brightgreen.svg)](https://github.com/DefTruth/lite.ai/tree/main/docs/hub/lite.ai.hub.onnx.md#lite.ai.hub.onnx-classification)  | *classification* | ✅ | [demo](https://github.com/DefTruth/lite.ai/blob/main/examples/lite/cv/test_lite_shufflenetv2.cpp) |
|[DenseNet121](https://pytorch.org/hub/pytorch_vision_densenet/)|30.7M|       [torchvision](https://github.com/pytorch/vision)       | 🔥🔥🔥↑ | [![](https://img.shields.io/badge/onnx-done-brightgreen.svg)](https://github.com/DefTruth/lite.ai/tree/main/docs/hub/lite.ai.hub.onnx.md#lite.ai.hub.onnx-classification)  | *classification* | ✅ | [demo](https://github.com/DefTruth/lite.ai/blob/main/examples/lite/cv/test_lite_densenet.cpp) |
|[GhostNet](https://pytorch.org/hub/pytorch_vision_ghostnet/)|20M|       [torchvision](https://github.com/pytorch/vision)       | 🔥🔥🔥↑ | [![](https://img.shields.io/badge/onnx-done-brightgreen.svg)](https://github.com/DefTruth/lite.ai/tree/main/docs/hub/lite.ai.hub.onnx.md#lite.ai.hub.onnx-classification)  | *classification* | ✅ | [demo](https://github.com/DefTruth/lite.ai/blob/main/examples/lite/cv/test_lite_ghostnet.cpp) |
|[HdrDNet](https://pytorch.org/hub/pytorch_vision_hardnet//)|13M|       [torchvision](https://github.com/pytorch/vision)       | 🔥🔥🔥↑ | [![](https://img.shields.io/badge/onnx-done-brightgreen.svg)](https://github.com/DefTruth/lite.ai/tree/main/docs/hub/lite.ai.hub.onnx.md#lite.ai.hub.onnx-classification) | *classification* | ✅ | [demo](https://github.com/DefTruth/lite.ai/blob/main/examples/lite/cv/test_lite_hardnet.cpp) |
|[IBNNet](https://pytorch.org/hub/pytorch_vision_ibnnet/)|97M|       [torchvision](https://github.com/pytorch/vision)       | 🔥🔥🔥↑ | [![](https://img.shields.io/badge/onnx-done-brightgreen.svg)](https://github.com/DefTruth/lite.ai/tree/main/docs/hub/lite.ai.hub.onnx.md#lite.ai.hub.onnx-classification)  | *classification* | ✅ | [demo](https://github.com/DefTruth/lite.ai/blob/main/examples/lite/cv/test_lite_ibnnet.cpp) |
|[MobileNetV2](https://pytorch.org/hub/pytorch_vision_mobilenet_v2/)|13M|       [torchvision](https://github.com/pytorch/vision)       | 🔥🔥🔥↑ | [![](https://img.shields.io/badge/onnx-done-brightgreen.svg)](https://github.com/DefTruth/lite.ai/tree/main/docs/hub/lite.ai.hub.onnx.md#lite.ai.hub.onnx-classification)  | *classification* | ✅ | [demo](https://github.com/DefTruth/lite.ai/blob/main/examples/lite/cv/test_lite_mobilenetv2.cpp) |
|[ResNet](https://pytorch.org/hub/pytorch_vision_resnet/)|44M|       [torchvision](https://github.com/pytorch/vision)       | 🔥🔥🔥↑ | [![](https://img.shields.io/badge/onnx-done-brightgreen.svg)](https://github.com/DefTruth/lite.ai/tree/main/docs/hub/lite.ai.hub.onnx.md#lite.ai.hub.onnx-classification)  | *classification* | ✅ | [demo](https://github.com/DefTruth/lite.ai/blob/main/examples/lite/cv/test_lite_resnet.cpp) |
|[ResNeXt](https://pytorch.org/hub/pytorch_vision_resnext/)|95M|       [torchvision](https://github.com/pytorch/vision)       | 🔥🔥🔥↑ | [![](https://img.shields.io/badge/onnx-done-brightgreen.svg)](https://github.com/DefTruth/lite.ai/tree/main/docs/hub/lite.ai.hub.onnx.md#lite.ai.hub.onnx-classification)  | *classification* | ✅ | [demo](https://github.com/DefTruth/lite.ai/blob/main/examples/lite/cv/test_lite_resnext.cpp) |


* Segmentation.   

|Class|Size|From|Awesome|File|Type|State|Usage|
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
|[DeepLabV3ResNet101](https://pytorch.org/hub/pytorch_vision_deeplabv3_resnet101/)|232M|       [torchvision](https://github.com/pytorch/vision)       | 🔥🔥🔥↑ | [![](https://img.shields.io/badge/onnx-done-brightgreen.svg)](https://github.com/DefTruth/lite.ai/tree/main/docs/hub/lite.ai.hub.onnx.md#lite.ai.hub.onnx-segmentation) | *segmentation* | ✅ | [demo](https://github.com/DefTruth/lite.ai/blob/main/examples/lite/cv/test_lite_deeplabv3_resnet101.cpp) |
|[FCNResNet101](https://pytorch.org/hub/pytorch_vision_fcn_resnet101/)|207M|       [torchvision](https://github.com/pytorch/vision)       | 🔥🔥🔥↑ | [![](https://img.shields.io/badge/onnx-done-brightgreen.svg)](https://github.com/DefTruth/lite.ai/tree/main/docs/hub/lite.ai.hub.onnx.md#lite.ai.hub.onnx-segmentation) | *segmentation* | ✅ | [demo](https://github.com/DefTruth/lite.ai/blob/main/examples/lite/cv/test_lite_fcn_resnet101.cpp) |


* Style Transfer.  

|Class|Size|From|Awesome|File|Type|State|Usage|
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
|[FastStyleTransfer](https://github.com/onnx/models/blob/master/vision/style_transfer/fast_neural_style)|6.4M|        [onnx-models](https://github.com/onnx/models)         | 🔥🔥🔥↑ | [![](https://img.shields.io/badge/onnx-done-brightgreen.svg)](https://github.com/DefTruth/lite.ai/tree/main/docs/hub/lite.ai.hub.onnx.md#lite.ai.hub.onnx-style-transfer) | *style* | ✅ | [demo](https://github.com/DefTruth/lite.ai/blob/main/examples/lite/cv/test_lite_fast_style_transfer.cpp) |


* Colorization.  

|Class|Size|From|Awesome|File|Type|State|Usage|
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
|[Colorizer](https://github.com/richzhang/colorization)|123M|  [colorization](https://github.com/richzhang/colorization)   | 🔥🔥🔥↑ | [![](https://img.shields.io/badge/onnx-done-brightgreen.svg)](https://github.com/DefTruth/lite.ai/tree/main/docs/hub/lite.ai.hub.onnx.md#lite.ai.hub.onnx-colorization) | *colorization* | ✅ | [demo](https://github.com/DefTruth/lite.ai/blob/main/examples/lite/cv/test_lite_colorizer.cpp) |  


* Super Resolution.  

|Class|Size|From|Awesome|File|Type|State|Usage|
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
|[SubPixelCNN](https://github.com/niazwazir/SUB_PIXEL_CNN)|234K| [...PIXEL...](https://github.com/niazwazir/SUB_PIXEL_CNN)  |    🔥↑    | [![](https://img.shields.io/badge/onnx-done-brightgreen.svg)](https://github.com/DefTruth/lite.ai/tree/main/docs/hub/lite.ai.hub.onnx.md#lite.ai.hub.onnx-super-resolution) | *resolution* | ✅ | [demo](https://github.com/DefTruth/lite.ai/blob/main/examples/lite/cv/test_lite_subpixel_cnn.cpp) |

</details>

## 4. Examples for Lite.AI.  

<div id="lite.ai-Examples-for-Lite.AI"></div>

More examples can be found at [lite.ai-demos](https://github.com/DefTruth/lite.ai/tree/main/examples/lite/cv).  Note that the default backend for Lite.AI is `onnxruntime`, for the reason that onnxruntime supports the most of onnx's operators. Click the Expand ▶️ button will show you more examples for the specific topic you are interested in.

<div id="lite.ai-object-detection"></div>

#### Example0: Object Detection using [YoloV5](https://github.com/ultralytics/yolov5). Download model from Model-Zoo[<sup>2</sup>](#lite.ai-2).
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

The output is:
<div align='center'>
  <img src='logs/test_lite_yolov5_1.jpg' height="256px">
  <img src='logs/test_lite_yolov5_2.jpg' height="256px">
</div>

Or you can use Newest 🔥🔥 ! YOLO series's detector [YOLOX](https://github.com/Megvii-BaseDetection/YOLOX) or [YoloR](https://github.com/WongKinYiu/yolor). They got the similar results.  

****  

<div id="lite.ai-face-alignment"></div>

#### Example1: 1000 Facial Landmarks Detection using [FaceLandmarks1000](https://github.com/Single430/FaceLandmark1000). Download model from Model-Zoo[<sup>2</sup>](#lite.ai-2).
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
The output is:
<div align='center'>
  <img src='logs/test_lite_face_landmarks_1000.jpg' height="224px" width="224px">
  <img src='logs/test_lite_face_landmarks_1000_2.jpg' height="224px" width="224px">
  <img src='logs/test_lite_face_landmarks_1000_0.jpg' height="224px" width="224px">
</div>    

****  

<div id="lite.ai-colorization"></div>

#### Example2: Colorization using [colorization](https://github.com/richzhang/colorization). Download model from Model-Zoo[<sup>2</sup>](#lite.ai-2).
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
The output is:

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

<div id="lite.ai-face-recognition"></div>  

#### Example3: Face Recognition using [ArcFace](https://github.com/deepinsight/insightface/tree/master/recognition/arcface_torch). Download model from Model-Zoo[<sup>2</sup>](#lite.ai-2).

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

The output is:
<div align='center'>
  <img src='examples/lite/resources/test_lite_arcface_resnet_0.png' height="224px" width="224px">
  <img src='examples/lite/resources/test_lite_arcface_resnet_1.png' height="224px" width="224px">
  <img src='examples/lite/resources/test_lite_arcface_resnet_2.png' height="224px" width="224px">
</div>  

> Detected Sim01: 0.721159  Sim02: -0.0626267  

****

<div id="lite.ai-face-detection"></div>

#### Example4: Face Detection using [UltraFace](https://github.com/Linzaer/Ultra-Light-Fast-Generic-Face-Detector-1MB). Download model from Model-Zoo[<sup>2</sup>](#lite.ai-2).
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
The output is:
<div align='center'>
  <img src='logs/test_lite_ultraface.jpg' height="224px" width="224px">
  <img src='logs/test_lite_ultraface_2.jpg' height="224px" width="224px">
  <img src='logs/test_lite_ultraface_3.jpg' height="224px" width="224px">
</div>  


<div id="lite.ai-segmentation"></div>  
<div id="lite.ai-face-attributes-analysis"></div>  
<div id="lite.ai-image-classification"></div>  
<div id="lite.ai-head-pose-estimation"></div>
<div id="lite.ai-style-transfer"></div>
<div id="lite.ai-matting"></div>

<details>
<summary> ⚠️ Expand All Examples for Each Topic in Lite.AI.</summary>  

<details>
<summary> 4.1 Expand Examples for Object Detection.</summary>

#### 4.1 Object Detection using [YoloV5](https://github.com/ultralytics/yolov5). Download model from Model-Zoo[<sup>2</sup>](#lite.ai-2).
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

The output is:
<div align='center'>
  <img src='logs/test_lite_yolov5_1.jpg' height="256px">
  <img src='logs/test_lite_yolov5_2.jpg' height="256px">
</div>

Or you can use Newest 🔥🔥 ! YOLO series's detector [YOLOX](https://github.com/Megvii-BaseDetection/YOLOX) . They got the similar results.  

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
The output is:
<div align='center'>
  <img src='logs/test_lite_yolox_1.jpg' height="256px">
  <img src='logs/test_lite_yolox_2.jpg' height="256px">
</div>    

More classes for general object detection.
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
```  

</details>


<details>
<summary> 4.2 Expand Examples for Face Recognition.</summary>

#### 4.2 Face Recognition using [ArcFace](https://github.com/deepinsight/insightface/tree/master/recognition/arcface_torch). Download model from Model-Zoo[<sup>2</sup>](#lite.ai-2).

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

The output is:
<div align='center'>
  <img src='examples/lite/resources/test_lite_arcface_resnet_0.png' height="224px" width="224px">
  <img src='examples/lite/resources/test_lite_arcface_resnet_1.png' height="224px" width="224px">
  <img src='examples/lite/resources/test_lite_arcface_resnet_2.png' height="224px" width="224px">
</div>  

> Detected Sim01: 0.721159  Sim02: -0.0626267  

More classes for face recognition.
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
<summary> 4.3 Expand Examples for Segmentation.</summary>

#### 4.3 Segmentation using [DeepLabV3ResNet101](https://pytorch.org/hub/pytorch_vision_deeplabv3_resnet101/). Download model from Model-Zoo[<sup>2</sup>](#lite.ai-2).
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

The output is:
<div align='center'>
  <img src='examples/lite/resources/test_lite_deeplabv3_resnet101.png' height="256px">
  <img src='logs/test_lite_deeplabv3_resnet101.jpg' height="256px">
</div> 

More classes for segmentation.
```c++
auto *segment = new lite::cv::segmentation::FCNResNet101(onnx_path);
```

</details>



<details>
<summary> 4.4 Expand Examples for Face Attributes Analysis.</summary>

#### 4.4 Age Estimation using [SSRNet](https://github.com/oukohou/SSR_Net_Pytorch) . Download model from Model-Zoo[<sup>2</sup>](#lite.ai-2).
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
The output is:
<div align='center'>
  <img src='logs/test_lite_ssrnet.jpg' height="224px" width="224px">
  <img src='logs/test_lite_gender_googlenet.jpg' height="224px" width="224px">
  <img src='logs/test_lite_emotion_ferplus.jpg' height="224px" width="224px">
</div>    

More classes for face attributes analysis.
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
<summary> 4.5 Expand Examples for Image Classification.</summary>

#### 4.5 1000 Classes Classification using [DenseNet](https://pytorch.org/hub/pytorch_vision_densenet/). Download model from Model-Zoo[<sup>2</sup>](#lite.ai-2).
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

The output is:
<div align='center'>
  <img src='examples/lite/resources/test_lite_densenet.jpg' height="224px" width="224px">
  <img src='logs/test_lite_densenet.png' height="224px" width="500px">
</div>  

More classes for image classification.
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
<summary> 4.6 Expand Examples for Face Detection.</summary>

#### 4.6 Face Detection using [UltraFace](https://github.com/Linzaer/Ultra-Light-Fast-Generic-Face-Detector-1MB). Download model from Model-Zoo[<sup>2</sup>](#lite.ai-2).
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
The output is:  
<div align='center'>
  <img src='logs/test_lite_ultraface.jpg' height="224px" width="224px">
  <img src='logs/test_lite_ultraface_2.jpg' height="224px" width="224px">
  <img src='logs/test_lite_ultraface_3.jpg' height="224px" width="224px">
</div>  

More classes for face detection.
```c++
auto *detector = new lite::face::detect::UltraFace(onnx_path);  // 1.1Mb only !
auto *detector = new lite::face::detect::FaceBoxes(onnx_path);  // 3.8Mb only ! 
auto *detector = new lite::face::detect::RetinaFace(onnx_path);  // 1.6Mb only ! CVPR2020
```

</details>


<details>
<summary> 4.7 Expand Examples for Colorization.</summary>

#### 4.7 Colorization using [colorization](https://github.com/richzhang/colorization). Download model from Model-Zoo[<sup>2</sup>](#lite.ai-2).
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
The output is:

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
<summary> 4.8 Expand Examples for Head Pose Estimation.</summary>

#### 4.8 Head Pose Estimation using [FSANet](https://github.com/omasaht/headpose-fsanet-pytorch). Download model from Model-Zoo[<sup>2</sup>](#lite.ai-2).

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

The output is:
<div align='center'>
  <img src='logs/test_lite_fsanet.jpg' height="224px" width="224px">
  <img src='logs/test_lite_fsanet_2.jpg' height="224px" width="224px">
  <img src='logs/test_lite_fsanet_3.jpg' height="224px" width="224px">
</div>  

</details>


<details>
<summary> 4.9 Expand Examples for Face Alignment.</summary>

#### 4.9 1000 Facial Landmarks Detection using [FaceLandmarks1000](https://github.com/Single430/FaceLandmark1000). Download model from Model-Zoo[<sup>2</sup>](#lite.ai-2).
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
The output is:
<div align='center'>
  <img src='logs/test_lite_face_landmarks_1000.jpg' height="224px" width="224px">
  <img src='logs/test_lite_face_landmarks_1000_2.jpg' height="224px" width="224px">
  <img src='logs/test_lite_face_landmarks_1000_0.jpg' height="224px" width="224px">
</div>    

More classes for face alignment.
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
<summary> 4.10 Expand Examples for Style Transfer.</summary>

#### 4.10 Style Transfer using [FastStyleTransfer](https://github.com/onnx/models/tree/master/vision/style_transfer/fast_neural_style). Download model from Model-Zoo[<sup>2</sup>](#lite.ai-2).
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
The output is:

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
<summary> 4.11 Expand Examples for Image Matting.</summary>

* *todo*⚠️

</details>

</details>


## 5. Lite.AI API Docs.

<div id="lite.ai-Lite.AI-API-Docs"></div>

### 5.1 Default Version APIs.  
More details of Default Version APIs can be found at [default-version-api-docs](https://github.com/DefTruth/lite.ai/blob/main/docs/api/default.md) . For examples, the interface for YoloV5 is:

> `lite::cv::detection::YoloV5`
```c++
void detect(const cv::Mat &mat, std::vector<types::Boxf> &detected_boxes, 
            float score_threshold = 0.25f, float iou_threshold = 0.45f,
            unsigned int topk = 100, unsigned int nms_type = NMS::OFFSET);
```


<details>
<summary> Expand for ONNXRuntime, MNN and NCNN version APIs.</summary>

### 5.2 ONNXRuntime Version APIs.  
More details of ONNXRuntime Version APIs can be found at [onnxruntime-version-api-docs](https://github.com/DefTruth/lite.ai/blob/main/docs/api/onnxruntime.md) . For examples, the interface for YoloV5 is:

> `lite::onnxruntime::cv::detection::YoloV5`
```c++
void detect(const cv::Mat &mat, std::vector<types::Boxf> &detected_boxes, 
            float score_threshold = 0.25f, float iou_threshold = 0.45f,
            unsigned int topk = 100, unsigned int nms_type = NMS::OFFSET);
```


### 5.3 MNN Version APIs. 

(*todo*⚠️: Not implementation now, coming soon.)  

> `lite::mnn::cv::detection::YoloV5`

> `lite::mnn::cv::detection::YoloV4`

> `lite::mnn::cv::detection::YoloV3`

> `lite::mnn::cv::detection::SSD`  

...


### 5.4 NCNN Version APIs.

(*todo*⚠️: Not implementation now, coming soon.)

> `lite::ncnn::cv::detection::YoloV5`

> `lite::ncnn::cv::detection::YoloV4`

> `lite::ncnn::cv::detection::YoloV3`

> `lite::ncnn::cv::detection::SSD`

...

</details>



## 6. Other Docs.  

<div id="lite.ai-Other-Docs"></div>  
<div id="lite.ai-1"></div>

<details>
<summary> Expand More Details for Other Docs.</summary>

### 6.1 Docs for ONNXRuntime. 
* [Rapid implementation of your inference using BasicOrtHandler](https://github.com/DefTruth/lite.ai/blob/main/docs/ort/ort_handler.zh.md)  
* [Some very useful onnxruntime c++ interfaces](https://github.com/DefTruth/lite.ai/blob/main/docs/ort/ort_useful_api.zh.md)  
* [How to compile a single model in this library you needed](https://github.com/DefTruth/lite.ai/blob/main/docs/ort/ort_build_single.zh.md)
* [How to convert SubPixelCNN to ONNX and implements with onnxruntime c++](https://github.com/DefTruth/lite.ai/blob/main/docs/ort/ort_subpixel_cnn.zh.md)
* [How to convert Colorizer to ONNX and implements with onnxruntime c++](https://github.com/DefTruth/lite.ai/blob/main/docs/ort/ort_colorizer.zh.md)
* [How to convert SSRNet to ONNX and implements with onnxruntime c++](https://github.com/DefTruth/lite.ai/blob/main/docs/ort/ort_ssrnet.zh.md)
* [How to convert YoloV3 to ONNX and implements with onnxruntime c++](https://github.com/DefTruth/lite.ai/blob/main/docs/ort/ort_yolov3.zh.md)
* [How to convert YoloV5 to ONNX and implements with onnxruntime c++](https://github.com/DefTruth/lite.ai/blob/main/docs/ort/ort_yolov5.zh.md)


### 6.2 Docs for [third_party](https://github.com/DefTruth/lite.ai/tree/main/third_party).  
Other build documents for different engines and different targets will be added later.


|Library|Target|Docs|
|:---:|:---:|:---:|
|OpenCV| mac-x86_64 | [opencv-mac-x86_64-build.zh.md](https://github.com/DefTruth/lite.ai/blob/main/docs/third_party/opencv-mac-x86_64-build.zh.md) |
|OpenCV| android-arm | [opencv-static-android-arm-build.zh.md](https://github.com/DefTruth/lite.ai/blob/main/docs/third_party/opencv-static-android-arm-build.zh.md) |
|onnxruntime| mac-x86_64 | [onnxruntime-mac-x86_64-build.zh.md](https://github.com/DefTruth/lite.ai/blob/main/docs/third_party/onnxruntime-mac-x86_64-build.zh.md) |
|onnxruntime| android-arm | [onnxruntime-android-arm-build.zh.md](https://github.com/DefTruth/lite.ai/blob/main/docs/third_party/onnxruntime-android-arm-build.zh.md) |
|NCNN| mac-x86_64 | todo⚠️ |
|MNN| mac-x86_64 | todo⚠️ |
|TNN| mac-x86_64 | todo⚠️ |

</details>


## 7. References.  

<div id="lite.ai-References"></div>

Many thanks to the following projects. All the Lite.AI's models are sourced from these repos. 

* [YOLOX]( https://github.com/Megvii-BaseDetection/YOLOX) (🔥🔥new!!↑)
* [insightface](https://github.com/deepinsight/insightface) (🔥🔥🔥↑)
* [yolov5](https://github.com/ultralytics/yolov5) (🔥🔥💥↑)  

<details>
<summary> Expand More Details for References.</summary>

* [headpose-fsanet-pytorch](https://github.com/omasaht/headpose-fsanet-pytorch) (🔥↑)
* [pfld_106_face_landmarks](https://github.com/Hsintao/pfld_106_face_landmarks) (🔥🔥↑)
* [Ultra-Light-Fast-Generic-Face-Detector-1MB](https://github.com/Linzaer/Ultra-Light-Fast-Generic-Face-Detector-1MB) (🔥🔥🔥↑)
* [onnx-models](https://github.com/onnx/models) (🔥🔥🔥↑)
* [SSR_Net_Pytorch](https://github.com/oukohou/SSR_Net_Pytorch) (🔥↑)
* [colorization](https://github.com/richzhang/colorization) (🔥🔥🔥↑)
* [SUB_PIXEL_CNN](https://github.com/niazwazir/SUB_PIXEL_CNN) (🔥↑)
* [YOLOv4-pytorch](https://github.com/argusswift/YOLOv4-pytorch) (🔥🔥🔥↑)
* [torchvision](https://github.com/pytorch/vision) (🔥🔥🔥↑)
* [facenet-pytorch](https://github.com/timesler/facenet-pytorch) (🔥↑)
* [face.evoLVe.PyTorch](https://github.com/ZhaoJ9014/face.evoLVe.PyTorch) (🔥🔥🔥↑)
* [TFace](https://github.com/Tencent/TFace) (🔥🔥↑)
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

Star 🌟👆🏻 this repo if it does any helps to you ~

<!----

## 8. Contributions.  
<div id="lite.ai-Contributions"></div>  

Do you want to contribute a model? To get started, just open an new issue with the title like *contribute-lite.ai-cv-detection-xxx(your model | repo name | alias)*, such as *contribute-lite.ai-cv-detection-YoloV5*. And, put the link to you project and point out how to find your model files and inference code. I will try to convert you model file and add it into *Lite.AI* as soon as I can. Then, I will list you repo in Lite.AI's Model Zoo and Acknowledgements. You could found a template at [contribute-issue-template](https://github.com/DefTruth/lite.ai/issues/1) . For examples:   

* issue name: contribute-lite.ai-cv-detection-YoloV5  
* model information: The information for the model is listed below.

| Project Address                                        | Author      | Model File                                                   | Inference                                                    |
| ------------------------------------------------------ | ----------- | ------------------------------------------------------------ | ------------------------------------------------------------ |
| [yolov5](https://github.com/ultralytics/yolov5) (🔥🔥💥↑) | ultralytics | [yolov5-model-pytorch-hub](https://github.com/ultralytics/yolov5/issues/36) | [detect.py](https://github.com/ultralytics/yolov5/blob/master/detect.py) |

Do you want a C++ user friendly version of you own pretrained models ? Come and join us ~  🙃🤪🍀

----->  

## License.  

<div id="lite.ai-License"></div>

Only the code of [Lite.AI](#lite.ai-Introduction) is released under the MIT License.

## Citations.

If you use this library in your project, please, cite it as follows.
```
@code{lite.ai2021,
  title={Lite.AI: A simple and user friendly C++ library of awesome AI models.},
  url={https://github.com/DefTruth/lite.ai},
  note={Open-source software available at https://github.com/DefTruth/lite.ai},
  author={Qiu},
  year={2021}
}
```
