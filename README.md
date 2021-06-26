

# litehub 🚀🚀🌟  

<div align='center'>
  <img src='logs/test_lite_yolov5_1.jpg' height="200px" width="200px">
  <img src='logs/test_lite_deeplabv3_resnet101.jpg' height="200px" width="200px">
  <img src='logs/test_lite_ssd_mobilenetv1.jpg' height="200px" width="200px">
  <img src='logs/test_lite_ultraface.jpg' height="200px" width="200px">
  <br> 
  <img src='logs/test_lite_pfld.jpg' height="200px" width="200px">
  <img src='logs/test_lite_fsanet.jpg' height="200px" width="200px">
  <img src='logs/test_lite_fast_style_transfer_candy.jpg' height="200px" width="200px">
  <img src='logs/test_lite_fast_style_transfer_mosaic.jpg' height="200px" width="200px"> 
</div>

*litehub* for onnxruntime/ncnn/mnn. This library integrates some interesting models and implement with onnxruntime/ncnn/mnn. Such as `YoloV5`、`UltraFace`、`PFLD`、`Colorization`、`FastStyleTransfer` and so on.
Most of the models come from `ONNX-Model-Zoo`, `PytorchHub` and `other open source projects`. All models used will be cited. Many thanks to these contributors. What you see is what you get, and hopefully you get something out of it.

## 1. Dependencies.  
* Mac OS.  

  install `opencv` and `onnxruntime` libraries using Homebrew.

```shell
  brew update
  brew install opencv
  brew install onnxruntime
```
​		or you can download the built dependencies from this repo. See [third_party](https://github.com/DefTruth/litehub/tree/main/third_party) and build-docs[<sup>1</sup>](#refer-anchor-1).
* Linux & Windows. (`TODO`)
* Inference Engine Plans:
  * Doing:
    * [x] `onnxruntime` 
  * TODO:
    * `NCNN`
    * `MNN`
    * `OpenMP` 

## 2. Build LiteHub.
* Build shared library for MacOS from sources or you can download the built lib from [liblitehub](https://github.com/DefTruth/litehub/tree/main/build/litehub/lib) (`TODO: Linux & Windows`). Note that litehub is only support for `onnxruntime` now.
```shell
git clone  --depth=1 https://github.com/DefTruth/litehub.git
cd litehub
sh ./build.sh
cd ./build/litehub/lib && otool -L liblitehub.dylib 
liblitehub.dylib:
        @rpath/liblitehub.dylib (compatibility version 0.0.0, current version 0.0.0)
        @rpath/libopencv_highgui.4.5.dylib (compatibility version 4.5.0, current version 4.5.2)
        @rpath/libonnxruntime.1.7.0.dylib (compatibility version 0.0.0, current version 1.7.0)
        ...
cd ../ && tree .
├── bin
├── include
│   ├── lite
│   │   ├── backend.h
│   │   ├── config.h
│   │   └── lite.h
│   └── ort
└── lib
    └── liblitehub.dylib
```
* Run built examples:  
```shell
cd ./build/litehub/bin && ls | grep lite
-rwxr-xr-x  1 root  staff   3.4M Jun 26 23:10 liblitehub.dylib
...
-rwxr-xr-x  1 root  staff   196K Jun 26 23:10 lite_yolov4
-rwxr-xr-x  1 root  staff   196K Jun 26 23:10 lite_yolov5
...

./lite_yolov5
LITEORT_DEBUG LogId: ../../../hub/onnx/cv/yolov5s.onnx
=============== Input-Dims ==============
...
detected num_anchors: 25200
generate_bboxes num: 66
Default Version Detected Boxes Num: 5
```

* Link `litehub` shared lib. You need to make sure that `OpenCV` and `onnxruntime` are linked correctly. Just like:

```cmake
# link opencv.
set(OpenCV_DIR ${THIRDPARTY_DIR}/opencv/4.5.2/x86_64/lib/cmake/opencv4)
find_package(OpenCV 4 REQUIRED)
if (OpenCV_FOUND)
   include_directories(${OpenCV_INCLUDE_DIRS})
   set(OpenCV_LIBS opencv_highgui opencv_core opencv_imgcodecs opencv_imgproc) # need only
else ()
   message(FATAL_ERROR "OpenCV library not found")
endif ()
# link onnxruntime.
set(ONNXRUNTIME_DIR ${THIRDPARTY_DIR}/onnxruntime/1.7.0/x86_64)
set(ONNXRUNTIME_INCLUDE_DIR ${ONNXRUNTIME_DIR}/include)
set(ONNXRUNTIME_LIBRARY_DIR ${ONNXRUNTIME_DIR}/lib)
include_directories(${ONNXRUNTIME_INCLUDE_DIR})
link_directories(${ONNXRUNTIME_LIBRARY_DIR})
# link litehub.
set(LITEHUB_DIR ${THIRDPARTY_DIR}/litehub)
set(LITEHUB_INCLUDE_DIR ${LITEHUB_DIR}/include)
set(LITEHUB_LIBRARY_DIR ${LITEHUB_DIR}/lib)
include_directories(${LITEHUB_INCLUDE_DIR})
link_directories(${LITEHUB_LIBRARY_DIR})
# add your executable
add_executable(executable_name test_executable_name.cpp)
target_link_libraries(executable_name litehub)  # link litehub
```

## 3. Model Zoo.

### 3.1 model-zoo for ONNX version.
Some of the models were converted by this repo, and others were referenced from third-party libraries.

<div id="refer-anchor-2"></div>

|Model|Size|Download|From|Type|Usage|
|:---:|:---:|:---:|:---:|:---:|:---:|
|[YoloV5](https://github.com/ultralytics/yolov5)|28Mb~335Mb|[Baidu Drive](https://pan.baidu.com/s/1X5y7bOSPyeBzT9nSgQiMIQ) code:g83e| [litehub](https://github.com/DefTruth/litehub/blob/main/docs/ort/ort_yolov5.zh.md) | *lite::cv::detectiion* | [demo](https://github.com/DefTruth/litehub/blob/main/examples/lite/cv/test_lite_yolov5.cpp) |
|[YoloV3](https://github.com/onnx/models/blob/master/vision/object_detection_segmentation/yolov3)|236Mb|[Baidu Drive](https://pan.baidu.com/s/1X5y7bOSPyeBzT9nSgQiMIQ) code:g83e| [YoloV3](https://github.com/onnx/models/blob/master/vision/object_detection_segmentation/yolov3) | *lite::cv::detectiion* | [demo](https://github.com/DefTruth/litehub/blob/main/examples/lite/cv/test_lite_yolov3.cpp) |
|[TinyYoloV3](https://github.com/onnx/models/blob/master/vision/object_detection_segmentation/tiny-yolov3)|33Mb|[Baidu Drive](https://pan.baidu.com/s/1X5y7bOSPyeBzT9nSgQiMIQ) code:g83e| [TinyYoloV3](https://github.com/onnx/models/blob/master/vision/object_detection_segmentation/tiny-yolov3) | *lite::cv::detectiion* | [demo](https://github.com/DefTruth/litehub/blob/main/examples/lite/cv/test_lite_tiny_yolov3.cpp) |
|[YoloV4](https://github.com/argusswift/YOLOv4-pytorch)|176Mb|[Baidu Drive](https://pan.baidu.com/s/1X5y7bOSPyeBzT9nSgQiMIQ) code:g83e| [litehub](https://github.com/DefTruth/litehub/blob/main/docs/ort/ort_yolov4.zh.md) | *lite::cv::detectiion* | [demo](https://github.com/DefTruth/litehub/blob/main/examples/lite/cv/test_lite_tiny_yolov4.cpp) |
|[SSD](https://github.com/onnx/models/blob/master/vision/object_detection_segmentation/ssd)|76Mb|[Baidu Drive](https://pan.baidu.com/s/1X5y7bOSPyeBzT9nSgQiMIQ) code:g83e| [SSD](https://github.com/onnx/models/blob/master/vision/object_detection_segmentation/ssd) | *lite::cv::detectiion* | [demo](https://github.com/DefTruth/litehub/blob/main/examples/lite/cv/test_lite_ssd.cpp) |
|[SSDMobileNetV1](https://github.com/onnx/models/blob/master/vision/object_detection_segmentation/ssd-mobilenetv1)|27Mb|[Baidu Drive](https://pan.baidu.com/s/1X5y7bOSPyeBzT9nSgQiMIQ) code:g83e| [SSDMobileNetV1](https://github.com/onnx/models/blob/master/vision/object_detection_segmentation/ssd-mobilenetv1) | *lite::cv::detectiion* | [demo](https://github.com/DefTruth/litehub/blob/main/examples/lite/cv/test_lite_ssd_mobilenetv1.cpp) |
|[EfficientNet-Lite4](https://github.com/onnx/models/blob/master/vision/classification/efficientnet-lite4)|49Mb|[Baidu Drive](https://pan.baidu.com/s/1X5y7bOSPyeBzT9nSgQiMIQ) code:g83e| [EfficientNet-Lite4](https://github.com/onnx/models/blob/master/vision/classification/efficientnet-lite4) | *lite::cv::classification* | [demo](https://github.com/DefTruth/litehub/blob/main/examples/lite/cv/test_lite_efficientnet_lite4.cpp) |
|[ShuffleNetV2](https://github.com/onnx/models/blob/master/vision/classification/shufflenet)|8.7Mb|[Baidu Drive](https://pan.baidu.com/s/1X5y7bOSPyeBzT9nSgQiMIQ) code:g83e| [ShuffleNetV2](https://github.com/onnx/models/blob/master/vision/classification/shufflenet) | *lite::cv::classification* | [demo](https://github.com/DefTruth/litehub/blob/main/examples/lite/cv/test_lite_shufflenetv2.cpp) |
|[FSANet](https://github.com/omasaht/headpose-fsanet-pytorch)|1.2Mb|[Baidu Drive](https://pan.baidu.com/s/1X5y7bOSPyeBzT9nSgQiMIQ) code:g83e| [FSANet](https://github.com/omasaht/headpose-fsanet-pytorch)| *lite::cv::face* | [demo](https://github.com/DefTruth/litehub/blob/main/examples/lite/cv/test_lite_fsanet.cpp) |
|[PFLD](https://github.com/Hsintao/pfld_106_face_landmarks)|1.0Mb~5.5Mb|[Baidu Drive](https://pan.baidu.com/s/1X5y7bOSPyeBzT9nSgQiMIQ) code:g83e| [PFLD](https://github.com/Hsintao/pfld_106_face_landmarks) | *lite::cv::face* | [demo](https://github.com/DefTruth/litehub/blob/main/examples/lite/cv/test_lite_pfld.cpp) |
|[UltraFace](https://github.com/Linzaer/Ultra-Light-Fast-Generic-Face-Detector-1MB)|1.1Mb~1.5Mb|[Baidu Drive](https://pan.baidu.com/s/1X5y7bOSPyeBzT9nSgQiMIQ) code:g83e| [UltraFace](https://github.com/Linzaer/Ultra-Light-Fast-Generic-Face-Detector-1MB) | *lite::cv::face* | [demo](https://github.com/DefTruth/litehub/blob/main/examples/lite/cv/test_lite_ultraface.cpp) |
|[AgeGoogleNet](https://github.com/onnx/models/tree/master/vision/body_analysis/age_gender)|23Mb|[Baidu Drive](https://pan.baidu.com/s/1X5y7bOSPyeBzT9nSgQiMIQ) code:g83e| [AgeGoogleNet](https://github.com/onnx/models/tree/master/vision/body_analysis/age_gender) | *lite::cv::face* | [demo](https://github.com/DefTruth/litehub/blob/main/examples/lite/cv/test_lite_age_googlenet.cpp) |
|[GenderGoogleNet](https://github.com/onnx/models/tree/master/vision/body_analysis/age_gender)|23Mb|[Baidu Drive](https://pan.baidu.com/s/1X5y7bOSPyeBzT9nSgQiMIQ) code:g83e| [GenderGoogleNet](https://github.com/onnx/models/tree/master/vision/body_analysis/age_gender) | *lite::cv::face* | [demo](https://github.com/DefTruth/litehub/blob/main/examples/lite/cv/test_lite_gender_googlenet.cpp) |
|[EmotionFerPlus](https://github.com/onnx/models/blob/master/vision/body_analysis/emotion_ferplus)|33Mb|[Baidu Drive](https://pan.baidu.com/s/1X5y7bOSPyeBzT9nSgQiMIQ) code:g83e| [EmotionFerPlus](https://github.com/onnx/models/blob/master/vision/body_analysis/emotion_ferplus) | *lite::cv::face* | [demo](https://github.com/DefTruth/litehub/blob/main/examples/lite/cv/test_lite_emotion_ferplus.cpp) |
|[VGG16Age](https://github.com/onnx/models/tree/master/vision/body_analysis/age_gender)|514Mb|[Baidu Drive](https://pan.baidu.com/s/1X5y7bOSPyeBzT9nSgQiMIQ) code:g83e| [VGG16Age](https://github.com/onnx/models/tree/master/vision/body_analysis/age_gender) | *lite::cv::face* | [demo](https://github.com/DefTruth/litehub/blob/main/examples/lite/cv/test_lite_vgg16_age.cpp) |
|[VGG16Gender](https://github.com/onnx/models/tree/master/vision/body_analysis/age_gender)|512Mb|[Baidu Drive](https://pan.baidu.com/s/1X5y7bOSPyeBzT9nSgQiMIQ) code:g83e| [VGG16Gender](https://github.com/onnx/models/tree/master/vision/body_analysis/age_gender) | *lite::cv::face* | [demo](https://github.com/DefTruth/litehub/blob/main/examples/lite/cv/test_lite_vgg16_gender.cpp) |
|[SSRNet](https://github.com/oukohou/SSR_Net_Pytorch)|190Kb|[Baidu Drive](https://pan.baidu.com/s/1X5y7bOSPyeBzT9nSgQiMIQ) code:g83e| [litehub](https://github.com/DefTruth/litehub/blob/main/docs/ort/ort_ssrnet.zh.md) | *lite::cv::face* | [demo](https://github.com/DefTruth/litehub/blob/main/examples/lite/cv/test_lite_ssrnet.cpp) |
|[FastStyleTransfer](https://github.com/onnx/models/blob/master/vision/style_transfer/fast_neural_style)|6.4Mb|[Baidu Drive](https://pan.baidu.com/s/1X5y7bOSPyeBzT9nSgQiMIQ) code:g83e| [FastStyleTransfer](https://github.com/onnx/models/blob/master/vision/style_transfer/fast_neural_style) | *lite::cv::style* | [demo](https://github.com/DefTruth/litehub/blob/main/examples/lite/cv/test_lite_fast_style_transfer.cpp) |
|[ArcFaceResNet](https://github.com/onnx/models/blob/master/vision/body_analysis/arcface)|249Mb|[Baidu Drive](https://pan.baidu.com/s/1X5y7bOSPyeBzT9nSgQiMIQ) code:g83e| [ArcFaceResNet](https://github.com/onnx/models/blob/master/vision/body_analysis/arcface) | *lite::cv::faceid* | [demo](https://github.com/DefTruth/litehub/blob/main/examples/lite/cv/test_lite_arcface_resnet.cpp) |
|[Colorizer](https://github.com/richzhang/colorization)|123Mb~130Mb|[Baidu Drive](https://pan.baidu.com/s/1X5y7bOSPyeBzT9nSgQiMIQ) code:g83e| [litehub](https://github.com/DefTruth/litehub/blob/main/docs/ort/ort_colorizer.zh.md) | *lite::cv::colorization* | [demo](https://github.com/DefTruth/litehub/blob/main/examples/lite/cv/test_lite_colorizer.cpp) |
|[SubPixelCNN](https://github.com/niazwazir/SUB_PIXEL_CNN)|234Kb|[Baidu Drive](https://pan.baidu.com/s/1X5y7bOSPyeBzT9nSgQiMIQ) code:g83e| [litehub](https://github.com/DefTruth/litehub/blob/main/docs/ort/ort_subpixel_cnn.zh.md) | *lite::cv::resolution* | [demo](https://github.com/DefTruth/litehub/blob/main/examples/lite/cv/test_lite_subpixel_cnn.cpp) |
|[DeepLabV3ResNet101](https://pytorch.org/hub/pytorch_vision_deeplabv3_resnet101/)|232Mb|[Baidu Drive](https://pan.baidu.com/s/1X5y7bOSPyeBzT9nSgQiMIQ) code:g83e| [litehub](https://github.com/DefTruth/litehub/blob/main/docs/ort/ort_deeplabv3_resnet101.zh.md) | *lite::cv::segmentation* | [demo](https://github.com/DefTruth/litehub/blob/main/examples/lite/cv/test_lite_deeplabv3_resnet101.cpp) |
|[DenseNet121](https://pytorch.org/hub/pytorch_vision_densenet/)|30.7Mb|[Baidu Drive](https://pan.baidu.com/s/1X5y7bOSPyeBzT9nSgQiMIQ) code:g83e| [litehub](https://github.com/DefTruth/litehub/blob/main/docs/ort/ort_densenet121.zh.md) | *lite::cv::classification* | [demo](https://github.com/DefTruth/litehub/blob/main/examples/lite/cv/test_lite_densenet.cpp) |
|[FCNResNet101](https://pytorch.org/hub/pytorch_vision_fcn_resnet101/)|207Mb|[Baidu Drive](https://pan.baidu.com/s/1X5y7bOSPyeBzT9nSgQiMIQ) code:g83e| [litehub](https://github.com/DefTruth/litehub/blob/main/docs/ort/ort_fcn_resnet101.zh.md) | *lite::cv::segmentation* | [demo](https://github.com/DefTruth/litehub/blob/main/examples/lite/cv/test_lite_fcn_resnet101.cpp) |
|[GhostNet](https://pytorch.org/hub/pytorch_vision_ghostnet/)|20Mb|[Baidu Drive](https://pan.baidu.com/s/1X5y7bOSPyeBzT9nSgQiMIQ) code:g83e| [litehub](https://github.com/DefTruth/litehub/blob/main/docs/ort/ort_ghostnet.zh.md) | *lite::cv::classification* | [demo](https://github.com/DefTruth/litehub/blob/main/examples/lite/cv/test_lite_ghostnet.cpp) |
|[HdrDNet](https://pytorch.org/hub/pytorch_vision_hardnet//)|13Mb|[Baidu Drive](https://pan.baidu.com/s/1X5y7bOSPyeBzT9nSgQiMIQ) code:g83e| [litehub](https://github.com/DefTruth/litehub/blob/main/docs/ort/ort_hardnet.zh.md) | *lite::cv::classification* | [demo](https://github.com/DefTruth/litehub/blob/main/examples/lite/cv/test_lite_hardnet.cpp) |
|[IBNNet](https://pytorch.org/hub/pytorch_vision_ibnnet/)|97Mb|[Baidu Drive](https://pan.baidu.com/s/1X5y7bOSPyeBzT9nSgQiMIQ) code:g83e| [litehub](https://github.com/DefTruth/litehub/blob/main/docs/ort/ort_ibnnet.zh.md) | *lite::cv::classification* | [demo](https://github.com/DefTruth/litehub/blob/main/examples/lite/cv/test_lite_ibnnet.cpp) |
|[MobileNetV2](https://pytorch.org/hub/pytorch_vision_mobilenet_v2/)|13Mb|[Baidu Drive](https://pan.baidu.com/s/1X5y7bOSPyeBzT9nSgQiMIQ) code:g83e| [litehub](https://github.com/DefTruth/litehub/blob/main/docs/ort/ort_mobilenetv2.zh.md) | *lite::cv::classification* | [demo](https://github.com/DefTruth/litehub/blob/main/examples/lite/cv/test_lite_mobilenetv2.cpp) |
|[ResNet](https://pytorch.org/hub/pytorch_vision_resnet/)|44Mb|[Baidu Drive](https://pan.baidu.com/s/1X5y7bOSPyeBzT9nSgQiMIQ) code:g83e| [litehub](https://github.com/DefTruth/litehub/blob/main/docs/ort/ort_resnet.zh.md) | *lite::cv::classification* | [demo](https://github.com/DefTruth/litehub/blob/main/examples/lite/cv/test_lite_resnet.cpp) |
|[ResNeXt](https://pytorch.org/hub/pytorch_vision_resnext/)|95Mb|[Baidu Drive](https://pan.baidu.com/s/1X5y7bOSPyeBzT9nSgQiMIQ) code:g83e| [litehub](https://github.com/DefTruth/litehub/blob/main/docs/ort/ort_resnext.zh.md) | *lite::cv::classification* | [demo](https://github.com/DefTruth/litehub/blob/main/examples/lite/cv/test_lite_resnext.cpp) |

## 4. Examples for LiteHub.

More examples can find at [lite-examples](https://github.com/DefTruth/litehub/tree/main/examples/lite/cv).  Note that the default backend for `litehub` is `onnxruntime`.
#### 4.1 Object detection using [YoloV5](https://github.com/ultralytics/yolov5). Download model from Model-Zoo[<sup>2</sup>](#refer-anchor-2).
```c++
#include "lite/lite.h"

static void test_default()
{
  std::string onnx_path = "../../../hub/onnx/cv/yolov5s.onnx";
  std::string test_img_path = "../../../examples/lite/resources/test_lite_yolov5_1.jpg";
  std::string save_img_path = "../../../logs/test_lite_yolov5_1.jpg";
  
  lite::cv::detection::YoloV5 *yolov5 = new lite::cv::detection::YoloV5(onnx_path); 
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

#### 4.2 Segmentation using [DeepLabV3ResNet101](https://pytorch.org/hub/pytorch_vision_deeplabv3_resnet101/). Download model from Model-Zoo[<sup>2</sup>](#refer-anchor-2).
```c++
#include "lite/lite.h"

static void test_default()
{
  std::string onnx_path = "../../../hub/onnx/cv/deeplabv3_resnet101_coco.onnx";
  std::string test_img_path = "../../../examples/lite/resources/test_lite_deeplabv3_resnet101.png";
  std::string save_img_path = "../../../logs/test_lite_deeplabv3_resnet101.jpg";

  lite::cv::segmentation::DeepLabV3ResNet101 *deeplabv3_resnet101 =
      new lite::cv::segmentation::DeepLabV3ResNet101(onnx_path, 16); // 16 threads

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

#### 4.3 Style transfer using [FastStyleTransfer](https://github.com/onnx/models/tree/master/vision/style_transfer/fast_neural_style). Download model from Model-Zoo[<sup>2</sup>](#refer-anchor-2).
```c++
#include "lite/lite.h"

static void test_default()
{
  std::string onnx_path = "../../../hub/onnx/cv/style-candy-8.onnx";
  std::string test_img_path = "../../../examples/lite/resources/test_lite_fast_style_transfer.jpg";
  std::string save_img_path = "../../../logs/test_lite_fast_style_transfer_candy.jpg";
  
  lite::cv::style::FastStyleTransfer *fast_style_transfer =
     new lite::cv::style::FastStyleTransfer(onnx_path);
 
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


#### 4.4 Colorization using [colorization](https://github.com/richzhang/colorization). Download model from Model-Zoo[<sup>2</sup>](#refer-anchor-2).
```c++
#include "lite/lite.h"

static void test_default()
{
  std::string onnx_path = "../../../hub/onnx/cv/eccv16-colorizer.onnx";
  std::string test_img_path = "../../../examples/lite/resources/test_lite_colorizer_1.jpg";
  std::string save_img_path = "../../../logs/test_lite_eccv16_colorizer_1.jpg";
  
  lite::cv::colorization::Colorizer *colorizer = new lite::cv::colorization::Colorizer(onnx_path);
  
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



#### 4.5 Facial Landmarks detection using [PFLD](https://github.com/Hsintao/pfld_106_face_landmarks). Download model from Model-Zoo[<sup>2</sup>](#refer-anchor-2).
```c++
#include "lite/lite.h"

static void test_default()
{
  std::string onnx_path = "../../../hub/onnx/cv/pfld-106-v3.onnx";
  std::string test_img_path = "../../../examples/lite/resources/test_lite_pfld.png";
  std::string save_img_path = "../../../logs/test_lite_pfld.jpg";

  lite::cv::face::PFLD *pfld = new lite::cv::face::PFLD(onnx_path);

  lite::cv::types::Landmarks landmarks;
  cv::Mat img_bgr = cv::imread(test_img_path);
  pfld->detect(img_bgr, landmarks);
  lite::cv::utils::draw_landmarks_inplace(img_bgr, landmarks);
  cv::imwrite(save_img_path, img_bgr);
  
  delete pfld;
}
```
The output is:  
<div align='center'>
  <img src='logs/test_lite_pfld.jpg' height="224px" width="224px">
  <img src='logs/test_lite_pfld_2.jpg' height="224px" width="224px">
  <img src='logs/test_lite_pfld_3.jpg' height="224px" width="224px">
</div>    

#### 4.6 Face detection using [UltraFace](https://github.com/Linzaer/Ultra-Light-Fast-Generic-Face-Detector-1MB). Download model from Model-Zoo[<sup>2</sup>](#refer-anchor-2).
```c++
#include "lite/lite.h"

static void test_default()
{
  std::string onnx_path = "../../../hub/onnx/cv/ultraface-rfb-640.onnx";
  std::string test_img_path = "../../../examples/lite/resources/test_lite_ultraface.jpg";
  std::string save_img_path = "../../../logs/test_lite_ultraface.jpg";

  lite::cv::face::UltraFace *ultraface = new lite::cv::face::UltraFace(onnx_path);

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

#### 4.7 1000 classes Classification using [DenseNet](https://pytorch.org/hub/pytorch_vision_densenet/). Download model from Model-Zoo[<sup>2</sup>](#refer-anchor-2).
```c++
#include "lite/lite.h"

static void test_default()
{
  std::string onnx_path = "../../../hub/onnx/cv/densenet121.onnx";
  std::string test_img_path = "../../../examples/lite/resources/test_lite_densenet.jpg";

  lite::cv::classification::DenseNet *densenet = new lite::cv::classification::DenseNet(onnx_path);

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



#### 4.8 HeadPose Estimation using [FSANet](https://github.com/omasaht/headpose-fsanet-pytorch). Download model from Model-Zoo[<sup>2</sup>](#refer-anchor-2).

```c++
#include "lite/lite.h"

static void test_default()
{
  std::string onnx_path = "../../../hub/onnx/cv/fsanet-var.onnx";
  std::string test_img_path = "../../../examples/lite/resources/test_lite_fsanet.jpg";
  std::string save_img_path = "../../../logs/test_lite_fsanet.jpg";

  lite::cv::face::FSANet *fsanet = new lite::cv::face::FSANet(onnx_path);
  cv::Mat img_bgr = cv::imread(test_img_path);
  lite::cv::types::EulerAngles euler_angles;
  fsanet->detect(img_bgr, euler_angles);
  
  if (euler_angles.flag)
  {
    lite::cv::utils::draw_axis_inplace(img_bgr, euler_angles);
    cv::imwrite(save_img_path, img_bgr);
    std::cout << euler_angles.yaw << euler_angles.pitch << euler_angles.roll << std::endl;
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


## 5. LiteHub API Docs.

### 5.1 Default Version APIs. (TODO)

* `lite::cv::detection::Yolo5`: 
* `lite::cv::detection::Yolo4`: 
* `lite::cv::detection::Yolo3`: 
* `lite::cv::detection::TinyYoloV3`: 
* `lite::cv::detection::SSD`: 
* `lite::cv::detection::SSDMobileNetV1`: 
* `lite::cv::face::FSANet`:
* `lite::cv::face::UltraFace`: 
* `lite::cv::face::PFLD`: 
* `lite::cv::face::AgeGoogleNet`: 
* `lite::cv::face::GenderGoogleNet`: 
* `lite::cv::face::VGG16Age`: 
* `lite::cv::face::VGG16Gender`: 
* `lite::cv::face::EmotionFerPlus`: 
* `lite::cv::face::SSRNet`: 
* `lite::cv::faceid::ArcFaceResNet`: 
* `lite::cv::segmentation::DeepLabV3ResNet101`: 
* `lite::cv::segmentation::FCNResNet101`: 
* `lite::cv::style::FastStyleTransfer`: 
* `lite::cv::colorization::Colorizer`: 
* `lite::cv::resolution::SubPixelCNN`: 
* `lite::cv::classification::EfficientNetLite4`: 
* `lite::cv::classification::ShuffleNetV2`: 
* `lite::cv::classification::DenseNet`: 
* `lite::cv::classification::GhostNet`: 
* `lite::cv::classification::HdrDNet`: 
* `lite::cv::classification::MobileNetV2`: 
* `lite::cv::classification::ResNet`: 
* `lite::cv::classification::ResNeXt`: 
* `lite::cv::utils::hard_nms:`
* `lite::cv::utils::blending_nms:`
* `lite::cv::utils::offset_nms:`

### 5.2 ONNXRuntime Version APIs.  (TODO)

* `lite::onnxruntime::cv::detection::Yolo5`: 
* `lite::onnxruntime::cv::detection::Yolo4`: 
* `lite::onnxruntime::cv::detection::Yolo3`: 
* `lite::onnxruntime::cv::detection::TinyYoloV3`: 
* `lite::onnxruntime::cv::detection::SSD`: 
* `lite::onnxruntime::cv::detection::SSDMobileNetV1`: 
* `lite::onnxruntime::cv::face::FSANet`:
* `lite::onnxruntime::cv::face::UltraFace`: 
* `lite::onnxruntime::cv::face::PFLD`: 
* `lite::onnxruntime::cv::face::AgeGoogleNet`: 
* `lite::onnxruntime::cv::face::GenderGoogleNet`: 
* `lite::onnxruntime::cv::face::VGG16Age`: 
* `lite::onnxruntime::cv::face::VGG16Gender`: 
* `lite::onnxruntime::cv::face::EmotionFerPlus`: 
* `lite::onnxruntime::cv::face::SSRNet`: 
* `lite::onnxruntime::cv::faceid::ArcFaceResNet`: 
* `lite::onnxruntime::cv::segmentation::DeepLabV3ResNet101`: 
* `lite::onnxruntime::cv::segmentation::FCNResNet101`: 
* `lite::onnxruntime::cv::style::FastStyleTransfer`: 
* `lite::onnxruntime::cv::colorization::Colorizer`: 
* `lite::onnxruntime::cv::resolution::SubPixelCNN`: 
* `lite::onnxruntime::cv::classification::EfficientNetLite4`: 
* `lite::onnxruntime::cv::classification::ShuffleNetV2`: 
* `lite::onnxruntime::cv::classification::DenseNet`: 
* `lite::onnxruntime::cv::classification::GhostNet`: 
* `lite::onnxruntime::cv::classification::HdrDNet`: 
* `lite::onnxruntime::cv::classification::MobileNetV2`: 
* `lite::onnxruntime::cv::classification::ResNet`: 
* `lite::onnxruntime::cv::classification::ResNeXt`: 

### 5.3 MNN Version APIs.

* TODO


## 6. Other Docs.  
### 6.1 ONNXRuntime Inference Engine. 
* [Rapid implementation of your inference using BasicOrtHandler](https://github.com/DefTruth/litehub/blob/main/docs/ort/ort_handler.zh.md)  
* [Some very useful onnxruntime c++ interfaces](https://github.com/DefTruth/litehub/blob/main/docs/ort/ort_useful_api.zh.md)  
* [How to compile a single model in this library you needed](https://github.com/DefTruth/litehub/blob/main/docs/ort/ort_build_single.zh.md)
* [How to convert SubPixelCNN to ONNX and implements with onnxruntime c++](https://github.com/DefTruth/litehub/blob/main/docs/ort/ort_subpixel_cnn.zh.md)
* [How to convert Colorizer to ONNX and implements with onnxruntime c++](https://github.com/DefTruth/litehub/blob/main/docs/ort/ort_colorizer.zh.md)
* [How to convert SSRNet to ONNX and implements with onnxruntime c++](https://github.com/DefTruth/litehub/blob/main/docs/ort/ort_ssrnet.zh.md)
* [How to convert YoloV3 to ONNX and implements with onnxruntime c++](https://github.com/DefTruth/litehub/blob/main/docs/ort/ort_yolov3.zh.md)
* [How to convert YoloV5 to ONNX and implements with onnxruntime c++](https://github.com/DefTruth/litehub/blob/main/docs/ort/ort_yolov5.zh.md)

### 6.2 How to build [third_party](https://github.com/DefTruth/litehub/tree/main/third_party).  
Other build documents for different engines and different targets will be added later.

<div id="refer-anchor-1"></div> 


|Library|Target|Docs|
|:---:|:---:|:---:|
|OpenCV| mac-x86_64 | [opencv-mac-x86_64-build.zh.md](https://github.com/DefTruth/litehub/blob/main/docs/third_party/opencv-mac-x86_64-build.zh.md) |
|OpenCV| android-arm | [opencv-static-android-arm-build.zh.md](https://github.com/DefTruth/litehub/blob/main/docs/third_party/opencv-static-android-arm-build.zh.md) |
|onnxruntime| mac-x86_64 | [onnxruntime-mac-x86_64-build.zh.md](https://github.com/DefTruth/litehub/blob/main/docs/third_party/onnxruntime-mac-x86_64-build.zh.md) |
|onnxruntime| android-arm | [onnxruntime-android-arm-build.zh.md](https://github.com/DefTruth/litehub/blob/main/docs/third_party/onnxruntime-android-arm-build.zh.md) |
|NCNN| mac-x86_64 | TODO |
|MNN| mac-x86_64 | TODO |
|TNN| mac-x86_64 | TODO |

