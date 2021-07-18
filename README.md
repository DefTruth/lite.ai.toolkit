

# Lite.AI 🚀🚀🌟  

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

## Introduction.    

<div id="refer-anchor-Introduction"></div> 

*Lite.AI* 🚀🚀🌟 is a simple, low-coupling, and user-friendly C++ library for awesome🔥🔥🔥 AI models, such as YoloV5, YoloV4, DeepLabV3, ArcFace, CosFace, Colorization, SSD, etc. And, it only relies on *[OpenCV](https://github.com/opencv/opencv)* and commonly used inference engines, namely, *[onnxruntime](https://github.com/microsoft/onnxruntime)*, *[ncnn](https://github.com/Tencent/ncnn)*, and *[mnn](https://github.com/alibaba/MNN)*. It currently mainly includes some *CV(Computer Vision 💻)* modules, such as [object detection](#refer-anchor-object-detection), [face detection](#refer-anchor-face-detection), [style transfer](#refer-anchor-style-transfer), [face alignment](#refer-anchor-face-alignment), [face recognition](#refer-anchor-face-recognition), [segmentation](#refer-anchor-segmentation), [colorization](#refer-anchor-colorization), [face attributes analysis](#refer-anchor-face-attributes-analysis), [image classification](#refer-anchor-image-classification), [matting](#refer-anchor-matting), etc.  You can use these awesome models through *lite::cv::Type::Model* syntax, such as *[lite::cv::detection::YoloV5](#refer-anchor-object-detection)* or *[lite::cv::face::detect::UltraFace](#refer-anchor-face-detection)*.  I do have plans to add *NLP* or *ASR* modules, but not coming soon. Currently, I am focusing🔍 on *Computer Vision* 💻 . It is important to note that the models here are all from third-party projects. All models used will be cited. Many thanks to these contributors. Have a good travel ~ 🙃🤪🍀  

****

* Important Note !!!   
  ⚠️ *Lite.AI* was rename from the *LiteHub* repo ! *LiteHub* will no longer be maintained.   
  
  
* Working on. 👇🏻
  * ✅ [object detection](#refer-anchor-object-detection) 
  * ✅ [image classification](#refer-anchor-object-detection) 
  * ❇️ [face detection](#refer-anchor-face-detection) 
  * ❇️ [face alignment](#refer-anchor-face-alignment) 
  * ❇️ [face recognition](#refer-anchor-face-recognition) 
  * ❇️ [face attributes analysis](#refer-anchor-face-attributes-analysis)
  * ⚠️ [segmentation](#refer-anchor-segmentation)
  * ⚠️ [style transfer](#refer-anchor-style-transfer)
  * ⚠️ [colorization](#refer-anchor-colorization)
  * ⚠️ [matting](#refer-anchor-matting)
  

****
<div id="refer-anchor-Related-Lite.AI-Projects"></div>

* Related Lite.AI Projects. 👇🏻
  * ❇️ [lite.ai](https://github.com/DefTruth/lite.ai) (*doing*✋🏻)
  * ❇️ [lite.ai-onnxruntime](https://github.com/DefTruth/lite.ai-onnxruntime) (*doing*✋🏻)
  * ⚠️ [lite.ai-mnn](https://github.com/DefTruth/lite.ai-mnn) (*todo*️)
  * ⚠️ [lite.ai-ncnn](https://github.com/DefTruth/lite.ai-ncnn) (*todo*️)
  * ❇️ [lite.ai-release](https://github.com/DefTruth/lite.ai-release) (*doing*✋🏻)
  * ⚠️ [lite.ai-python](https://github.com/DefTruth/lite.ai-python) (*todo*️)
  * ⚠️ [lite.ai-jni](https://github.com/DefTruth/lite.ai-jni) (*todo*️)  
  
****

## License.  

<div id="refer-anchor-Related-License"></div>

The code of [Lite.AI](#refer-anchor-Introduction) is released under the MIT License.

****

## Contents.
* [Introduction](#refer-anchor-Introduction)
* [Related Lite.AI Projects](#refer-anchor-Related-Lite.AI-Projects)
* [Dependencies](#refer-anchor-Dependencies)
* [Model Zoo](#refer-anchor-Model-Zoo)
* [Build Lite.AI](#refer-anchor-Build-Lite.AI)
* [Examples for LiteHub](#refer-anchor-Examples-for-Lite.AI)
* [LiteHub API Docs](#refer-anchor-Lite.AI-API-Docs)
* [Other Docs](#refer-anchor-Other-Docs)
* [Acknowledgements](#refer-anchor-Acknowledgements)
* [License](#refer-anchor-License)
* [Contributions](#refer-anchor-Contributions)

****

## 1. Dependencies.  

<div id="refer-anchor-Dependencies"></div>

* Mac OS.  
install `OpenCV` and `onnxruntime` libraries using Homebrew or you can download the built dependencies from this repo. See [third_party](https://github.com/DefTruth/litehub/tree/main/third_party) and build-docs[<sup>1</sup>](#refer-anchor-1) for more details.

```shell
  brew update
  brew install opencv
  brew install onnxruntime
```

* Linux & Windows. (*todo*⚠️)
* Inference Engine Plans:
  * *doing*:  
    ❇️ `onnxruntime` 
  * *todo*:  
    ⚠️ `NCNN`  
    ⚠️ `MNN`  
    ⚠️ `OpenMP`

****

## 2. Model Zoo.

<div id="refer-anchor-Model-Zoo"></div>

### 2.1 Namespace and Lite.AI modules.

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

### 2.2 Models for ONNX version.

Most of the models were converted by Lite.AI, and others were referenced from third-party libraries. The name of the class here will be different from the original repository, because different repositories have different implementations of the same algorithm. For example, ArcFace in [insightface](https://github.com/deepinsight/insightface) is different from ArcFace in [face.evoLVe.PyTorch](https://github.com/ZhaoJ9014/face.evoLVe.PyTorch) . ArcFace in [insightface](https://github.com/deepinsight/insightface) uses Arc-Loss + Softmax, while ArcFace in [face.evoLVe.PyTorch](https://github.com/ZhaoJ9014/face.evoLVe.PyTorch) uses Arc-Loss + Focal-Loss. Lite.AI uses naming to make the necessary distinctions between models from different sources.  Therefore, in Lite.AI, different names of the same algorithm mean that the corresponding models come from different repositories, different implementations, or use different training data, etc. Just jump to [lite.ai-demos](https://github.com/DefTruth/lite.ai/tree/main/examples/lite/cv) to figure out the usage of each model in Lite.AI. ✅ means passed the test and ⚠️ means not implements yet but coming soon. For models which denoted ✅, you can use it through *lite::cv::Type::Model* syntax, such as *[lite::cv::detection::YoloV5](#refer-anchor-object-detection)* or *[lite::cv::face::detect::UltraFace](#refer-anchor-face-detection)*. More details can be found at [Examples for Lite.AI](#refer-anchor-Examples-for-Lite.AI) .  
([Baidu Drive](https://pan.baidu.com/s/1elUGcx7CZkkjEoYhTMwTRQ) code: 8gin) <div id="refer-anchor-2"></div>

|Model|Size|From|Awesome|File|Type|State|Usage|
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
|[YoloV5](https://github.com/ultralytics/yolov5)|28M|[yolov5](https://github.com/ultralytics/yolov5)|🔥🔥💥↑| [lite.ai](https://github.com/DefTruth/lite.ai/blob/main/docs/ort/ort_yolov5.zh.md) | *detection* | ✅ | [demo](https://github.com/DefTruth/lite.ai/blob/main/examples/lite/cv/test_lite_yolov5.cpp) |
|[YoloV3](https://github.com/onnx/models/blob/master/vision/object_detection_segmentation/yolov3)|236M|[onnx-models](https://github.com/onnx/models)|🔥🔥🔥↑| - | *detection* | ✅ | [demo](https://github.com/DefTruth/lite.ai/blob/main/examples/lite/cv/test_lite_yolov3.cpp) |
|[TinyYoloV3](https://github.com/onnx/models/blob/master/vision/object_detection_segmentation/tiny-yolov3)|33M|        [onnx-models](https://github.com/onnx/models)         | 🔥🔥🔥↑ | - | *detection* | ✅ | [demo](https://github.com/DefTruth/lite.ai/blob/main/examples/lite/cv/test_lite_tiny_yolov3.cpp) |
|[YoloV4](https://github.com/argusswift/YOLOv4-pytorch)|176M| [YOLOv4...](https://github.com/argusswift/YOLOv4-pytorch) | 🔥🔥🔥↑ | [lite.ai](https://github.com/DefTruth/lite.ai/blob/main/docs/ort/ort_yolov4.zh.md) | *detection* | ✅ | [demo](https://github.com/DefTruth/lite.ai/blob/main/examples/lite/cv/test_lite_tiny_yolov4.cpp) |
|[SSD](https://github.com/onnx/models/blob/master/vision/object_detection_segmentation/ssd)|76M|        [onnx-models](https://github.com/onnx/models)         | 🔥🔥🔥↑ | - | *detection* | ✅ | [demo](https://github.com/DefTruth/lite.ai/blob/main/examples/lite/cv/test_lite_ssd.cpp) |
|[SSDMobileNetV1](https://github.com/onnx/models/blob/master/vision/object_detection_segmentation/ssd-mobilenetv1)|27M|        [onnx-models](https://github.com/onnx/models)         | 🔥🔥🔥↑ | - | *detection* | ✅ | [demo](https://github.com/DefTruth/lite.ai/blob/main/examples/lite/cv/test_lite_ssd_mobilenetv1.cpp) |
|[EfficientNet-Lite4](https://github.com/onnx/models/blob/master/vision/classification/efficientnet-lite4)|49M|        [onnx-models](https://github.com/onnx/models)         | 🔥🔥🔥↑ | - | *classification* | ✅ | [demo](https://github.com/DefTruth/lite.ai/blob/main/examples/lite/cv/test_lite_efficientnet_lite4.cpp) |
|[ShuffleNetV2](https://github.com/onnx/models/blob/master/vision/classification/shufflenet)|8.7M|        [onnx-models](https://github.com/onnx/models)         | 🔥🔥🔥↑ | - | *classification* | ✅ | [demo](https://github.com/DefTruth/lite.ai/blob/main/examples/lite/cv/test_lite_shufflenetv2.cpp) |
|[FSANet](https://github.com/omasaht/headpose-fsanet-pytorch)|1.2M| [...fsanet...](https://github.com/omasaht/headpose-fsanet-pytorch) | 🔥↑ | - | *face::pose* | ✅ | [demo](https://github.com/DefTruth/lite.ai/blob/main/examples/lite/cv/test_lite_fsanet.cpp) |
|[PFLD](https://github.com/Hsintao/pfld_106_face_landmarks)|1.0M| [pfld_106_...](https://github.com/Hsintao/pfld_106_face_landmarks) |  🔥🔥↑  | - | *face::align* | ✅ | [demo](https://github.com/DefTruth/lite.ai/blob/main/examples/lite/cv/test_lite_pfld.cpp) |
|[UltraFace](https://github.com/Linzaer/Ultra-Light-Fast-Generic-Face-Detector-1MB)|1.1M| [Ultra-Light...](https://github.com/Linzaer/Ultra-Light-Fast-Generic-Face-Detector-1MB) | 🔥🔥🔥↑ | - | *face::detect* | ✅ | [demo](https://github.com/DefTruth/lite.ai/blob/main/examples/lite/cv/test_lite_ultraface.cpp) |
|[AgeGoogleNet](https://github.com/onnx/models/tree/master/vision/body_analysis/age_gender)|23M|        [onnx-models](https://github.com/onnx/models)         | 🔥🔥🔥↑ | - | *face::attr* | ✅ | [demo](https://github.com/DefTruth/lite.ai/blob/main/examples/lite/cv/test_lite_age_googlenet.cpp) |
|[GenderGoogleNet](https://github.com/onnx/models/tree/master/vision/body_analysis/age_gender)|23M|        [onnx-models](https://github.com/onnx/models)         | 🔥🔥🔥↑ | - | *face::attr* | ✅ | [demo](https://github.com/DefTruth/lite.ai/blob/main/examples/lite/cv/test_lite_gender_googlenet.cpp) |
|[EmotionFerPlus](https://github.com/onnx/models/blob/master/vision/body_analysis/emotion_ferplus)|33M|        [onnx-models](https://github.com/onnx/models)         | 🔥🔥🔥↑ | - | *face::attr* | ✅ | [demo](https://github.com/DefTruth/lite.ai/blob/main/examples/lite/cv/test_lite_emotion_ferplus.cpp) |
|[VGG16Age](https://github.com/onnx/models/tree/master/vision/body_analysis/age_gender)|514M|        [onnx-models](https://github.com/onnx/models)         | 🔥🔥🔥↑ | - | *face::attr* | ✅ | [demo](https://github.com/DefTruth/lite.ai/blob/main/examples/lite/cv/test_lite_vgg16_age.cpp) |
|[VGG16Gender](https://github.com/onnx/models/tree/master/vision/body_analysis/age_gender)|512M|        [onnx-models](https://github.com/onnx/models)         | 🔥🔥🔥↑ | - | *face::attr* | ✅ | [demo](https://github.com/DefTruth/lite.ai/blob/main/examples/lite/cv/test_lite_vgg16_gender.cpp) |
|[SSRNet](https://github.com/oukohou/SSR_Net_Pytorch)|190K| [SSR_Net...](https://github.com/oukohou/SSR_Net_Pytorch) | 🔥↑ | [lite.ai](https://github.com/DefTruth/lite.ai/blob/main/docs/ort/ort_ssrnet.zh.md) | *face::attr* | ✅ | [demo](https://github.com/DefTruth/lite.ai/blob/main/examples/lite/cv/test_lite_ssrnet.cpp) |
|[FastStyleTransfer](https://github.com/onnx/models/blob/master/vision/style_transfer/fast_neural_style)|6.4M|        [onnx-models](https://github.com/onnx/models)         | 🔥🔥🔥↑ | - | *style* | ✅ | [demo](https://github.com/DefTruth/lite.ai/blob/main/examples/lite/cv/test_lite_fast_style_transfer.cpp) |
|[ArcFaceResNet](https://github.com/deepinsight/insightface/tree/master/recognition/arcface_torch)|92M|  [insightface](https://github.com/deepinsight/insightface)   | 🔥🔥🔥↑ | [lite.ai](https://github.com/DefTruth/lite.ai/) | *faceid* | ✅ | [demo](https://github.com/DefTruth/lite.ai/blob/main/examples/lite/cv/test_lite_arcface_resnet.cpp) |
|[GlintCosFace](https://github.com/deepinsight/insightface/tree/master/recognition/arcface_torch)|92M|  [insightface](https://github.com/deepinsight/insightface)   | 🔥🔥🔥↑ | [lite.ai](https://github.com/DefTruth/lite.ai/) | *faceid* | ✅ | [demo](https://github.com/DefTruth/lite.ai/blob/main/examples/lite/cv/test_lite_glint_cosface.cpp) |
|[GlintPartialFC](https://github.com/deepinsight/insightface/tree/master/recognition/partial_fc)|170M|  [insightface](https://github.com/deepinsight/insightface)   | 🔥🔥🔥↑ | [lite.ai](https://github.com/DefTruth/lite.ai/) | *faceid* | ✅ | [demo](https://github.com/DefTruth/lite.ai/blob/main/examples/lite/cv/test_lite_glint_partial_fc.cpp) |
|[FaceNet](https://github.com/timesler/facenet-pytorch)|93M| [facenet...](https://github.com/timesler/facenet-pytorch) | 🔥🔥🔥↑ | [lite.ai](https://github.com/DefTruth/lite.ai/) | *faceid* | ✅ | [demo](https://github.com/DefTruth/lite.ai/blob/main/examples/lite/cv/test_lite_facenet.cpp) |
|[FocalArcFace](https://github.com/ZhaoJ9014/face.evoLVe.PyTorch)|166M| [face.evoLVe...](https://github.com/ZhaoJ9014/face.evoLVe.PyTorch) | 🔥🔥🔥↑ | [lite.ai](https://github.com/DefTruth/lite.ai/) | *faceid* | ✅ | [demo](https://github.com/DefTruth/lite.ai/blob/main/examples/lite/cv/test_lite_focal_arcface.cpp) |
|[FocalAsiaArcFace](https://github.com/ZhaoJ9014/face.evoLVe.PyTorch)|166M| [face.evoLVe...](https://github.com/ZhaoJ9014/face.evoLVe.PyTorch) | 🔥🔥🔥↑ | [lite.ai](https://github.com/DefTruth/lite.ai/) | *faceid* | ✅ | [demo](https://github.com/DefTruth/lite.ai/blob/main/examples/lite/cv/test_lite_focal_asia_arcface.cpp) |
|[TencentCurricularFace](https://github.com/Tencent/TFace/tree/master/tasks/distfc)|249M|          [TFace](https://github.com/Tencent/TFace)           |  🔥🔥↑  | [lite.ai](https://github.com/DefTruth/lite.ai/) | *faceid* | ✅ | [demo](https://github.com/DefTruth/lite.ai/blob/main/examples/lite/cv/test_lite_tencent_curricular_face.cpp) |
|[TencentCifpFace](https://github.com/Tencent/TFace/tree/master/tasks/cifp)|130M|          [TFace](https://github.com/Tencent/TFace)           |  🔥🔥↑  | [lite.ai](https://github.com/DefTruth/lite.ai/) | *faceid* | ✅ | [demo](https://github.com/DefTruth/lite.ai/blob/main/examples/lite/cv/test_lite_tencent_cifp_face.cpp) |
|[CenterLossFace](https://github.com/louis-she/center-loss.pytorch)| 280M |  [center-loss...](https://github.com/louis-she/center-loss.pytorch)           |  🔥🔥↑  | [lite.ai](https://github.com/DefTruth/lite.ai/) | *faceid* | ✅ | [demo](https://github.com/DefTruth/lite.ai/blob/main/examples/lite/cv/test_lite_center_loss_face.cpp) |
|[SphereFace](https://github.com/clcarwin/sphereface_pytorch)| 80M |  [sphere...](https://github.com/clcarwin/sphereface_pytorch)   |  🔥🔥↑  | [lite.ai](https://github.com/DefTruth/lite.ai/) | *faceid* | ✅️ | [demo](https://github.com/DefTruth/lite.ai/blob/main/examples/lite/cv/test_lite_sphere_face.cpp) |
|[PoseRobustFace](https://github.com/penincillin/DREAM)| 92M | [DREAM](https://github.com/penincillin/DREAM)  |  🔥🔥↑  | [lite.ai](https://github.com/DefTruth/lite.ai/) | *faceid* | ✅️ | [demo](https://github.com/DefTruth/lite.ai/blob/main/examples/lite/cv/test_lite_pose_robust_face.cpp) |
|[NaivePoseRobustFace](https://github.com/penincillin/DREAM)| 43M | [DREAM](https://github.com/penincillin/DREAM)  |  🔥🔥↑  | [lite.ai](https://github.com/DefTruth/lite.ai/) | *faceid* | ✅️ | [demo](https://github.com/DefTruth/lite.ai/blob/main/examples/lite/cv/test_lite_naive_pose_robust_face.cpp) |
|[MobileFaceNet](https://github.com/Xiaoccer/MobileFaceNet_Pytorch)| 3.8M |  [MobileFace...](https://github.com/Xiaoccer/MobileFaceNet_Pytorch)           |  🔥🔥↑  | [lite.ai](https://github.com/DefTruth/lite.ai/) | *faceid* | ✅ | [demo](https://github.com/DefTruth/lite.ai/blob/main/examples/lite/cv/test_lite_mobile_facenet.cpp) |
|[CavaGhostArcFace](https://github.com/cavalleria/cavaface.pytorch)| - | [cavaface...](https://github.com/cavalleria/cavaface.pytorch) |  🔥🔥↑  | [lite.ai](https://github.com/DefTruth/lite.ai/) | *faceid* | ⚠️ | - |
|[CavaCombinedFace](https://github.com/cavalleria/cavaface.pytorch)| - | [cavaface...](https://github.com/cavalleria/cavaface.pytorch) |  🔥🔥↑  | [lite.ai](https://github.com/DefTruth/lite.ai/) | *faceid* | ⚠️ | - |
|[Colorizer](https://github.com/richzhang/colorization)|123M|  [colorization](https://github.com/richzhang/colorization)   | 🔥🔥🔥↑ | [lite.ai](https://github.com/DefTruth/lite.ai/blob/main/docs/ort/ort_colorizer.zh.md) | *colorization* | ✅ | [demo](https://github.com/DefTruth/lite.ai/blob/main/examples/lite/cv/test_lite_colorizer.cpp) |
|[SubPixelCNN](https://github.com/niazwazir/SUB_PIXEL_CNN)|234K| [...PIXEL...](https://github.com/niazwazir/SUB_PIXEL_CNN)  |    🔥↑    | [lite.ai](https://github.com/DefTruth/lite.ai/blob/main/docs/ort/ort_subpixel_cnn.zh.md) | *resolution* | ✅ | [demo](https://github.com/DefTruth/lite.ai/blob/main/examples/lite/cv/test_lite_subpixel_cnn.cpp) |
|[DeepLabV3ResNet101](https://pytorch.org/hub/pytorch_vision_deeplabv3_resnet101/)|232M|       [torchvision](https://github.com/pytorch/vision)       | 🔥🔥🔥↑ | [lite.ai](https://github.com/DefTruth/lite.ai/blob/main/docs/ort/ort_deeplabv3_resnet101.zh.md) | *segmentation* | ✅ | [demo](https://github.com/DefTruth/lite.ai/blob/main/examples/lite/cv/test_lite_deeplabv3_resnet101.cpp) |
|[DenseNet121](https://pytorch.org/hub/pytorch_vision_densenet/)|30.7M|       [torchvision](https://github.com/pytorch/vision)       | 🔥🔥🔥↑ | [lite.ai](https://github.com/DefTruth/lite.ai/blob/main/docs/ort/ort_densenet121.zh.md) | *classification* | ✅ | [demo](https://github.com/DefTruth/lite.ai/blob/main/examples/lite/cv/test_lite_densenet.cpp) |
|[FCNResNet101](https://pytorch.org/hub/pytorch_vision_fcn_resnet101/)|207M|       [torchvision](https://github.com/pytorch/vision)       | 🔥🔥🔥↑ | [lite.ai](https://github.com/DefTruth/lite.ai/blob/main/docs/ort/ort_fcn_resnet101.zh.md) | *segmentation* | ✅ | [demo](https://github.com/DefTruth/lite.ai/blob/main/examples/lite/cv/test_lite_fcn_resnet101.cpp) |
|[GhostNet](https://pytorch.org/hub/pytorch_vision_ghostnet/)|20M|       [torchvision](https://github.com/pytorch/vision)       | 🔥🔥🔥↑ | [lite.ai](https://github.com/DefTruth/lite.ai/blob/main/docs/ort/ort_ghostnet.zh.md) | *classification* | ✅ | [demo](https://github.com/DefTruth/lite.ai/blob/main/examples/lite/cv/test_lite_ghostnet.cpp) |
|[HdrDNet](https://pytorch.org/hub/pytorch_vision_hardnet//)|13M|       [torchvision](https://github.com/pytorch/vision)       | 🔥🔥🔥↑ | [lite.ai](https://github.com/DefTruth/lite.ai/blob/main/docs/ort/ort_hardnet.zh.md) | *classification* | ✅ | [demo](https://github.com/DefTruth/lite.ai/blob/main/examples/lite/cv/test_lite_hardnet.cpp) |
|[IBNNet](https://pytorch.org/hub/pytorch_vision_ibnnet/)|97M|       [torchvision](https://github.com/pytorch/vision)       | 🔥🔥🔥↑ | [lite.ai](https://github.com/DefTruth/lite.ai/blob/main/docs/ort/ort_ibnnet.zh.md) | *classification* | ✅ | [demo](https://github.com/DefTruth/lite.ai/blob/main/examples/lite/cv/test_lite_ibnnet.cpp) |
|[MobileNetV2](https://pytorch.org/hub/pytorch_vision_mobilenet_v2/)|13M|       [torchvision](https://github.com/pytorch/vision)       | 🔥🔥🔥↑ | [lite.ai](https://github.com/DefTruth/lite.ai/blob/main/docs/ort/ort_mobilenetv2.zh.md) | *classification* | ✅ | [demo](https://github.com/DefTruth/lite.ai/blob/main/examples/lite/cv/test_lite_mobilenetv2.cpp) |
|[ResNet](https://pytorch.org/hub/pytorch_vision_resnet/)|44M|       [torchvision](https://github.com/pytorch/vision)       | 🔥🔥🔥↑ | [lite.ai](https://github.com/DefTruth/lite.ai/blob/main/docs/ort/ort_resnet.zh.md) | *classification* | ✅ | [demo](https://github.com/DefTruth/lite.ai/blob/main/examples/lite/cv/test_lite_resnet.cpp) |
|[ResNeXt](https://pytorch.org/hub/pytorch_vision_resnext/)|95M|       [torchvision](https://github.com/pytorch/vision)       | 🔥🔥🔥↑ | [lite.ai](https://github.com/DefTruth/lite.ai/blob/main/docs/ort/ort_resnext.zh.md) | *classification* | ✅ | [demo](https://github.com/DefTruth/lite.ai/blob/main/examples/lite/cv/test_lite_resnext.cpp) |
|[CategoryFaceChannel](https://github.com/pablovin/FaceChannel)|-|       [FaceChannel](https://github.com/pablovin/FaceChannel)       | 🔥↑ | [lite.ai](https://github.com/DefTruth/lite.ai/) | *face::attr* |  ⚠️ | - |
|[DemensionFaceChannel](https://github.com/pablovin/FaceChannel)|-|       [FaceChannel](https://github.com/pablovin/FaceChannel)       | 🔥↑ | [lite.ai](https://github.com/DefTruth/lite.ai/) | *face::attr* | ⚠️ | - |

Correspondence between the classes in *Lite.AI* and pretrained model files can be found at [lite.ai.hub.md](https://github.com/DefTruth/lite.ai/tree/main/lite.ai.hub.md). For examples, the pretrained model files for *lite::cv::detection::YoloV5* are listed as following.  

|             Model             | Pretrained ONNX files |                   Rename or Converted From (Repo)                   | Size  |
| :---------------------------: | :-------------------: | :----------------------------------------------------: | :---: |
| *lite::cv::detection::YoloV5* |     Yolov5l.onnx      | [yolov5](https://github.com/ultralytics/yolov5) (🔥🔥💥↑) | 188Mb |
| *lite::cv::detection::YoloV5* |     Yolov5m.onnx      | [yolov5](https://github.com/ultralytics/yolov5) (🔥🔥💥↑) | 85Mb  |
| *lite::cv::detection::YoloV5* |     Yolov5s.onnx      | [yolov5](https://github.com/ultralytics/yolov5) (🔥🔥💥↑) | 29Mb  |
| *lite::cv::detection::YoloV5* |     Yolov5x.onnx      | [yolov5](https://github.com/ultralytics/yolov5) (🔥🔥💥↑) | 351Mb |

It means that you can load the any one `yolov5*.onnx` according to your application through the same Lite.AI class *YoloV5*, such as:  

```c++
auto *yolov5 = new lite::cv::detection::YoloV5("yolov5x.onnx");  // for server
auto *yolov5 = new lite::cv::detection::YoloV5("yolov5l.onnx"); 
auto *yolov5 = new lite::cv::detection::YoloV5("yolov5m.onnx");  // for mobile device
auto *yolov5 = new lite::cv::detection::YoloV5("yolov5s.onnx"); 
```

****

<details>
<summary> Expand for the pretrianed models of MNN and NCNN version.</summary>

### 2.3 Models for MNN version.  
* *todo*⚠️  

****

### 2.4 Models for NCNN version.  
* *todo*⚠️

</details>

****

## 3. Build Lite.AI.  

<div id="refer-anchor-Build-Lite.AI"></div>

Build the shared lib of Lite.AI for MacOS from sources or you can download the built lib from [liblite.ai.dylib|so](https://github.com/DefTruth/lite.ai/tree/main/build/lite.ai/lib) (`TODO: Linux & Windows`). Note that Lite.AI uses `onnxruntime` as default backend, for the reason that onnxruntime supports the most of onnx's operators. For Linux and Windows, you need to build the shared libs of `OpenCV` and `onnxruntime` firstly and put then into the `third_party` directory. Please reference the build-docs[<sup>1</sup>](#refer-anchor-1) for `third_party`.  

* Clone the Lite.AI from sources:  
```shell
git clone --depth=1 -b v0.0.1 https://github.com/DefTruth/lite.ai.git  # stable
git clone --depth=1 https://github.com/DefTruth/lite.ai.git  # latest
```
* For users in China, you can try:
```shell
git clone --depth=1 -b v0.0.1 https://github.com.cnpmjs.org/DefTruth/lite.ai.git  # stable
git clone --depth=1 https://github.com.cnpmjs.org/DefTruth/lite.ai.git  # latest
```
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

****

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

* Link `lite.ai` shared lib. You need to make sure that `OpenCV` and `onnxruntime` are linked correctly. Just like:

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

****

## 4. Examples for Lite.AI.  

<div id="refer-anchor-Examples-for-Lite.AI"></div>

More examples can be found at [lite.ai-demos](https://github.com/DefTruth/lite.ai/tree/main/examples/lite/cv).  Note that the default backend for Lite.AI is `onnxruntime`, for the reason that onnxruntime supports the most of onnx's operators.

<div id="refer-anchor-object-detection"></div>

#### 4.1 Object Detection using [YoloV5](https://github.com/ultralytics/yolov5). Download model from Model-Zoo[<sup>2</sup>](#refer-anchor-2).
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

More models for general object detection.  
```c++
auto *detector = new lite::cv::detection::YoloV4(onnx_path); 
auto *detector = new lite::cv::detection::YoloV3(onnx_path); 
auto *detector = new lite::cv::detection::TinyYoloV3(onnx_path); 
auto *detector = new lite::cv::detection::SSD(onnx_path); 
auto *detector = new lite::cv::detection::SSDMobileNetV1(onnx_path); 
```

****

<div id="refer-anchor-segmentation"></div>

#### 4.2 Segmentation using [DeepLabV3ResNet101](https://pytorch.org/hub/pytorch_vision_deeplabv3_resnet101/). Download model from Model-Zoo[<sup>2</sup>](#refer-anchor-2).
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

More models for segmentation.
```c++
auto *segment = new lite::cv::segmentation::FCNResNet101(onnx_path);
```

****

<div id="refer-anchor-face-attributes-analysis"></div>

#### 4.3 Age Estimation using [SSRNet](https://github.com/oukohou/SSR_Net_Pytorch) . Download model from Model-Zoo[<sup>2</sup>](#refer-anchor-2).
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

More models for face attributes analysis.
```c++
auto *attribute = new lite::cv::face::attr::AgeGoogleNet(onnx_path);  
auto *attribute = new lite::cv::face::attr::GenderGoogleNet(onnx_path); 
auto *attribute = new lite::cv::face::attr::EmotionFerPlus(onnx_path);
auto *attribute = new lite::cv::face::attr::VGG16Age(onnx_path);
auto *attribute = new lite::cv::face::attr::VGG16Gender(onnx_path);
auto *attribute = new lite::cv::face::attr::CategoryFaceChannel(onnx_path); // emotion 
auto *attribute = new lite::cv::face::attr::DemensionFaceChannel(onnx_path); // emotion 
```

****

<div id="refer-anchor-image-classification"></div>

#### 4.4 1000 Classes Classification using [DenseNet](https://pytorch.org/hub/pytorch_vision_densenet/). Download model from Model-Zoo[<sup>2</sup>](#refer-anchor-2).
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

More models for image classification.
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

****

<div id="refer-anchor-face-recognition"></div>

#### 4.5 Face Recognition using [ArcFace](https://github.com/deepinsight/insightface/tree/master/recognition/arcface_torch). Download model from Model-Zoo[<sup>2</sup>](#refer-anchor-2).

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

More models for face recognition.
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
auto *recognition = new lite::cv::faceid::MobileFaceNet(onnx_path);
auto *recognition = new lite::cv::faceid::CavaGhostArcFace(onnx_path);
auto *recognition = new lite::cv::faceid::CavaCombinedFace(onnx_path);
```

****

<div id="refer-anchor-face-detection"></div>

<details>
<summary> 4.6 Expand Examples for Face Detection.</summary>

#### 4.6 Face Detection using [UltraFace](https://github.com/Linzaer/Ultra-Light-Fast-Generic-Face-Detector-1MB). Download model from Model-Zoo[<sup>2</sup>](#refer-anchor-2).
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

</details>

****

<div id="refer-anchor-colorization"></div>

<details>
<summary> 4.7 Expand Examples for Colorization.</summary>

#### 4.7 Colorization using [colorization](https://github.com/richzhang/colorization). Download model from Model-Zoo[<sup>2</sup>](#refer-anchor-2).
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

****

<div id="refer-anchor-head-pose-estimation"></div>

<details>
<summary> 4.8 Expand Examples for Head Pose Estimation.</summary>

#### 4.8 Head Pose Estimation using [FSANet](https://github.com/omasaht/headpose-fsanet-pytorch). Download model from Model-Zoo[<sup>2</sup>](#refer-anchor-2).

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

****

<div id="refer-anchor-face-alignment"></div>

<details>
<summary> 4.9 Expand Examples for Face Alignment.</summary>

#### 4.9 Facial Landmarks Detection using [PFLD](https://github.com/Hsintao/pfld_106_face_landmarks). Download model from Model-Zoo[<sup>2</sup>](#refer-anchor-2).
```c++
#include "lite/lite.h"

static void test_default()
{
  std::string onnx_path = "../../../hub/onnx/cv/pfld-106-v3.onnx";
  std::string test_img_path = "../../../examples/lite/resources/test_lite_pfld.png";
  std::string save_img_path = "../../../logs/test_lite_pfld.jpg";

  auto *pfld = new lite::cv::face::align::PFLD(onnx_path);

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

</details>

****

<div id="refer-anchor-style-transfer"></div>

<details>
<summary> 4.10 Expand Examples for Style Transfer.</summary>

#### 4.10 Style Transfer using [FastStyleTransfer](https://github.com/onnx/models/tree/master/vision/style_transfer/fast_neural_style). Download model from Model-Zoo[<sup>2</sup>](#refer-anchor-2).
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

****

<div id="refer-anchor-matting"></div>

<details>
<summary> 4.11 Expand Examples for Image Matting.</summary>

* *todo*⚠️

</details>

****


## 5. Lite.AI API Docs.

<div id="refer-anchor-Lite.AI-API-Docs"></div>

### 5.1 Default Version APIs.  
More details of Default Version APIs can be found at [default-version-api-docs](https://github.com/DefTruth/lite.ai/blob/main/docs/api/default.md) . For examples, the interface for YoloV5 is:

> `lite::cv::detection::YoloV5`
```c++
void detect(const cv::Mat &mat, std::vector<types::Boxf> &detected_boxes, 
            float score_threshold = 0.25f, float iou_threshold = 0.45f,
            unsigned int topk = 100, unsigned int nms_type = NMS::OFFSET);
```

****

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

****

### 5.3 MNN Version APIs. 

(*todo*⚠️: Not implementation now, coming soon.)  

> `lite::mnn::cv::detection::YoloV5`

> `lite::mnn::cv::detection::YoloV4`

> `lite::mnn::cv::detection::YoloV3`

> `lite::mnn::cv::detection::SSD`  

...

****

### 5.4 NCNN Version APIs.

(*todo*⚠️: Not implementation now, coming soon.)

> `lite::ncnn::cv::detection::YoloV5`

> `lite::ncnn::cv::detection::YoloV4`

> `lite::ncnn::cv::detection::YoloV3`

> `lite::ncnn::cv::detection::SSD`

...

</details>

****


## 6. Other Docs.  

<div id="refer-anchor-Other-Docs"></div>  
<div id="refer-anchor-1"></div> 

****

<details>
<summary> Expand for more details of Other Docs.</summary>

### 6.1 Docs for ONNXRuntime. 
* [Rapid implementation of your inference using BasicOrtHandler](https://github.com/DefTruth/lite.ai/blob/main/docs/ort/ort_handler.zh.md)  
* [Some very useful onnxruntime c++ interfaces](https://github.com/DefTruth/lite.ai/blob/main/docs/ort/ort_useful_api.zh.md)  
* [How to compile a single model in this library you needed](https://github.com/DefTruth/lite.ai/blob/main/docs/ort/ort_build_single.zh.md)
* [How to convert SubPixelCNN to ONNX and implements with onnxruntime c++](https://github.com/DefTruth/lite.ai/blob/main/docs/ort/ort_subpixel_cnn.zh.md)
* [How to convert Colorizer to ONNX and implements with onnxruntime c++](https://github.com/DefTruth/lite.ai/blob/main/docs/ort/ort_colorizer.zh.md)
* [How to convert SSRNet to ONNX and implements with onnxruntime c++](https://github.com/DefTruth/lite.ai/blob/main/docs/ort/ort_ssrnet.zh.md)
* [How to convert YoloV3 to ONNX and implements with onnxruntime c++](https://github.com/DefTruth/lite.ai/blob/main/docs/ort/ort_yolov3.zh.md)
* [How to convert YoloV5 to ONNX and implements with onnxruntime c++](https://github.com/DefTruth/lite.ai/blob/main/docs/ort/ort_yolov5.zh.md)

****

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

****

## 7. Acknowledgements.  

<div id="refer-anchor-Acknowledgements"></div>

Many thanks to the following projects. All the Lite.AI's models are sourced from these repos. Just jump to and star 🌟👉🏻 the any awesome one you are interested in ! Have a good travel ~ 🙃🤪🍀  

* [1] [headpose-fsanet-pytorch](https://github.com/omasaht/headpose-fsanet-pytorch) (🔥↑)
* [2] [pfld_106_face_landmarks](https://github.com/Hsintao/pfld_106_face_landmarks) (🔥🔥↑)
* [3] [Ultra-Light-Fast-Generic-Face-Detector-1MB](https://github.com/Linzaer/Ultra-Light-Fast-Generic-Face-Detector-1MB) (🔥🔥🔥↑)
* [4] [onnx-models](https://github.com/onnx/models) (🔥🔥🔥↑)
* [5] [SSR_Net_Pytorch](https://github.com/oukohou/SSR_Net_Pytorch) (🔥↑)
* [6] [insightface](https://github.com/deepinsight/insightface) (🔥🔥🔥↑)
* [7] [colorization](https://github.com/richzhang/colorization) (🔥🔥🔥↑)
* [8] [SUB_PIXEL_CNN](https://github.com/niazwazir/SUB_PIXEL_CNN) (🔥↑)
* [9] [YOLOv4-pytorch](https://github.com/argusswift/YOLOv4-pytorch) (🔥🔥🔥↑)
* [10] [yolov5](https://github.com/ultralytics/yolov5) (🔥🔥💥↑)
* [11] [torchvision](https://github.com/pytorch/vision) (🔥🔥🔥↑)
* [12] [facenet-pytorch](https://github.com/timesler/facenet-pytorch) (🔥↑)
* [13] [face.evoLVe.PyTorch](https://github.com/ZhaoJ9014/face.evoLVe.PyTorch) (🔥🔥🔥↑)
* [14] [TFace](https://github.com/Tencent/TFace) (🔥🔥↑)
* [15] [center-loss.pytorch](https://github.com/louis-she/center-loss.pytorch) (🔥🔥↑)
* [16] [sphereface_pytorch](https://github.com/clcarwin/sphereface_pytorch) (🔥🔥↑)
* [17] [DREAM](https://github.com/penincillin/DREAM) (🔥🔥↑)
* [18] [MobileFaceNet_Pytorch](https://github.com/Xiaoccer/MobileFaceNet_Pytorch) (🔥🔥↑)
* [19] [cavaface.pytorch](https://github.com/cavalleria/cavaface.pytorch) (🔥🔥↑)
* [20] [CurricularFace](https://github.com/HuangYG123/CurricularFace) (🔥🔥↑)
* [21] [FaceChannel](https://github.com/pablovin/FaceChannel) (🔥↑)
* [??] [lite.ai](https://github.com/DefTruth/lite.ai) ( 👈🏻 yet, I guess you might be also interested in this repo ~ 🙃🤪🍀)

****

## 8. Contributions.  
<div id="refer-anchor-Contributions"></div>  

Do you want to contribute a model? To get started, just open an new issue with the title like *contribute-lite.ai-cv-detection-xxx(your model | repo name | alias)*, such as *contribute-lite.ai-cv-detection-YoloV5*. And, put the link to you project and point out how to find your model files and inference code. I will try to convert you model file and add it into *Lite.AI* as soon as I can. Then, I will list you repo in Lite.AI's Model Zoo and Acknowledgements. You could found a template at [contribute-issue-template](https://github.com/DefTruth/lite.ai/issues/1) . For examples:   

* issue name: contribute-lite.ai-cv-detection-YoloV5  
* model information: The information for the model is listed below.

| Project Address                                        | Author      | Model File                                                   | Inference                                                    |
| ------------------------------------------------------ | ----------- | ------------------------------------------------------------ | ------------------------------------------------------------ |
| [yolov5](https://github.com/ultralytics/yolov5) (🔥🔥💥↑) | ultralytics | [yolov5-model-pytorch-hub](https://github.com/ultralytics/yolov5/issues/36) | [detect.py](https://github.com/ultralytics/yolov5/blob/master/detect.py) |

Do you want a C++ user friendly version of you own pretrained models ? Come and join us ~  🙃🤪🍀

