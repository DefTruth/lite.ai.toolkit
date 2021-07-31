# Lite.AI.Hub.ONNX 

Correspondence between the classes in *Lite.AI* and pretrained model files can be found at this document. For examples, the pretrained model files for *lite::cv::detection::YoloV5* and *lite::cv::detection::YoloX* are listed as follows.

|             Class             | Pretrained ONNX Files |               Rename or Converted From (Repo)                | Size  |
| :---------------------------: | :-------------------: | :----------------------------------------------------------: | :---: |
| *lite::cv::detection::YoloV5* |     yolov5l.onnx      |    [yolov5](https://github.com/ultralytics/yolov5) (🔥🔥💥↑)    | 188Mb |
| *lite::cv::detection::YoloV5* |     yolov5m.onnx      |    [yolov5](https://github.com/ultralytics/yolov5) (🔥🔥💥↑)    | 85Mb  |
| *lite::cv::detection::YoloV5* |     yolov5s.onnx      |    [yolov5](https://github.com/ultralytics/yolov5) (🔥🔥💥↑)    | 29Mb  |
| *lite::cv::detection::YoloV5* |     yolov5x.onnx      |    [yolov5](https://github.com/ultralytics/yolov5) (🔥🔥💥↑)    | 351Mb |
| *lite::cv::detection::YoloX*  |     yolox_x.onnx      | [YOLOX](https://github.com/Megvii-BaseDetection/YOLOX) (🔥🔥!!↑) | 378Mb |
| *lite::cv::detection::YoloX*  |     yolox_l.onnx      | [YOLOX](https://github.com/Megvii-BaseDetection/YOLOX) (🔥🔥!!↑) | 207Mb |
| *lite::cv::detection::YoloX*  |     yolox_m.onnx      | [YOLOX](https://github.com/Megvii-BaseDetection/YOLOX) (🔥🔥!!↑) | 97Mb  |
| *lite::cv::detection::YoloX*  |     yolox_s.onnx      | [YOLOX](https://github.com/Megvii-BaseDetection/YOLOX) (🔥🔥!!↑) | 34Mb  |
| *lite::cv::detection::YoloX*  |    yolox_tiny.onnx    | [YOLOX](https://github.com/Megvii-BaseDetection/YOLOX) (🔥🔥!!↑) | 19Mb  |
| *lite::cv::detection::YoloX*  |    yolox_nano.onnx    | [YOLOX](https://github.com/Megvii-BaseDetection/YOLOX) (🔥🔥!!↑) | 3.5Mb |

It means that you can load the any one `yolov5*.onnx` and  `yolox_*.onnx` according to your application through the same Lite.AI classes, such as *YoloV5*, *YoloX*, etc.

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

You can download all the pretrained models files of ONNX format from ([Baidu Drive](https://pan.baidu.com/s/1elUGcx7CZkkjEoYhTMwTRQ) code: 8gin) 

## Object Detection.  

<div id="lite.ai.hub.onnx-object-detection"></div>

|                 Class                 |      Pretrained ONNX Files      |              Rename or Converted From (Repo)              | Size  |
| :-----------------------------------: | :-----------------------------: | :-------------------------------------------------------: | :---: |
|     *lite::cv::detection::YoloV5*     |          yolov5l.onnx           |      [yolov5](https://github.com/ultralytics/yolov5)      | 188Mb |
|     *lite::cv::detection::YoloV5*     |          yolov5m.onnx           |      [yolov5](https://github.com/ultralytics/yolov5)      | 85Mb  |
|     *lite::cv::detection::YoloV5*     |          yolov5s.onnx           |      [yolov5](https://github.com/ultralytics/yolov5)      | 29Mb  |
|     *lite::cv::detection::YoloV5*     |          yolov5x.onnx           |      [yolov5](https://github.com/ultralytics/yolov5)      | 351Mb |
|     *lite::cv::detection::YoloX*      |          yolox_x.onnx           |  [YOLOX](https://github.com/Megvii-BaseDetection/YOLOX)   | 378Mb |
|     *lite::cv::detection::YoloX*      |          yolox_l.onnx           |  [YOLOX](https://github.com/Megvii-BaseDetection/YOLOX)   | 207Mb |
|     *lite::cv::detection::YoloX*      |          yolox_m.onnx           |  [YOLOX](https://github.com/Megvii-BaseDetection/YOLOX)   | 97Mb  |
|     *lite::cv::detection::YoloX*      |          yolox_s.onnx           |  [YOLOX](https://github.com/Megvii-BaseDetection/YOLOX)   | 34Mb  |
|     *lite::cv::detection::YoloX*      |         yolox_tiny.onnx         |  [YOLOX](https://github.com/Megvii-BaseDetection/YOLOX)   | 19Mb  |
|     *lite::cv::detection::YoloX*      |         yolox_nano.onnx         |  [YOLOX](https://github.com/Megvii-BaseDetection/YOLOX)   | 3.5Mb |
|     *lite::cv::detection::YoloV3*     |         yolov3-10.onnx          |       [onnx-models](https://github.com/onnx/models)       | 236Mb |
|   *lite::cv::detection::TinyYoloV3*   |       tiny-yolov3-11.onnx       |       [onnx-models](https://github.com/onnx/models)       | 33Mb  |
|     *lite::cv::detection::YoloV4*     | voc-mobilenetv2-yolov4-640.onnx | [YOLOv4...](https://github.com/argusswift/YOLOv4-pytorch) | 176Mb |
|     *lite::cv::detection::YoloV4*     | voc-mobilenetv2-yolov4-416.onnx | [YOLOv4...](https://github.com/argusswift/YOLOv4-pytorch) | 176Mb |
|      *lite::cv::detection::SSD*       |           ssd-10.onnx           |       [onnx-models](https://github.com/onnx/models)       | 76Mb  |
| *lite::cv::detection::SSDMobileNetV1* |    ssd_mobilenet_v1_10.onnx     |       [onnx-models](https://github.com/onnx/models)       | 27Mb  |


## Classification.  

<div id="lite.ai.hub.onnx-classification"></div>


|                    Class                     |   Pretrained ONNX Files    |         Rename or Converted From (Repo)          | Size  |
| :------------------------------------------: | :------------------------: | :----------------------------------------------: | :---: |
| *lite::cv::classification:EfficientNetLite4* | efficientnet-lite4-11.onnx |  [onnx-models](https://github.com/onnx/models)   | 49Mb  |
|   *lite::cv::classification::ShuffleNetV2*   |   shufflenet-v2-10.onnx    |  [onnx-models](https://github.com/onnx/models)   | 8.7Mb |
|   *lite::cv::classification::DenseNet121*    |      densenet121.onnx      | [torchvision](https://github.com/pytorch/vision) | 30Mb  |
|     *lite::cv::classification::GhostNet*     |       ghostnet.onnx        | [torchvision](https://github.com/pytorch/vision) | 20Mb  |
|     *lite::cv::classification::HdrDNet*      |        hardnet.onnx        | [torchvision](https://github.com/pytorch/vision) | 13Mb  |
|      *lite::cv::classification::IBNNet*      |       ibnnet18.onnx        | [torchvision](https://github.com/pytorch/vision) | 97Mb  |
|   *lite::cv::classification::MobileNetV2*    |      mobilenetv2.onnx      | [torchvision](https://github.com/pytorch/vision) | 13Mb  |
|      *lite::cv::classification::ResNet*      |       resnet18.onnx        | [torchvision](https://github.com/pytorch/vision) | 44Mb  |
|     *lite::cv::classification::ResNeXt*      |        resnext.onnx        | [torchvision](https://github.com/pytorch/vision) | 95Mb  |


## Face Detection.

<div id="lite.ai.hub.onnx-face-detection"></div>  

|                Class                | Pretrained ONNX Files  |               Rename or Converted From (Repo)                | Size  |
| :---------------------------------: | :--------------------: | :----------------------------------------------------------: | :---: |
| *lite::cv::face::detect::UltraFace* | ultraface-rfb-640.onnx | [Ultra-Light...](https://github.com/Linzaer/Ultra-Light-Fast-Generic-Face-Detector-1MB) | 1.5Mb |
| *lite::cv::face::detect::UltraFace* | ultraface-rfb-320.onnx | [Ultra-Light...](https://github.com/Linzaer/Ultra-Light-Fast-Generic-Face-Detector-1MB) | 1.2Mb |
| *lite::cv::face::detect::RetinaFace* | Pytorch_RetinaFace_resnet50.onnx | [...Retinaface](https://github.com/biubug6/Pytorch_Retinaface)| 104Mb |
| *lite::cv::face::detect::RetinaFace* | Pytorch_RetinaFace_mobile0.25.onnx | [...Retinaface](https://github.com/biubug6/Pytorch_Retinaface) | 1.6Mb |



## Face Alignment.  

<div id="lite.ai.hub.onnx-face-alignment"></div>  


|             Class             | Pretrained ONNX Files |               Rename or Converted From (Repo)                | Size  |
| :---------------------------: | :-------------------: | :----------------------------------------------------------: | :---: |
| *lite::cv::face::align::PFLD* |  pfld-106-lite.onnx   | [pfld_106_...](https://github.com/Hsintao/pfld_106_face_landmarks) | 1.0Mb |
| *lite::cv::face::align::PFLD* |   pfld-106-v3.onnx    | [pfld_106_...](https://github.com/Hsintao/pfld_106_face_landmarks) | 5.5Mb |
| *lite::cv::face::align::PFLD* |   pfld-106-v2.onnx    | [pfld_106_...](https://github.com/Hsintao/pfld_106_face_landmarks) | 5.0Mb |
| *lite::cv::face::align::PFLD98* |   PFLD-pytorch-pfld.onnx  | [PFLD...](https://github.com/polarisZhao/PFLD-pytorch) | 4.8Mb |
| *lite::cv::face::align::MobileNetV268* |   pytorch_face_landmarks_landmark_detection_56.onnx  | [...landmark](https://github.com/cunjian/pytorch_face_landmark) | 9.4Mb |
| *lite::cv::face::align::MobileNetV2SE68* |   pytorch_face_landmarks_landmark_detection_56_se_external.onnx  | [...landmark](https://github.com/cunjian/pytorch_face_landmark) | 11Mb |
| *lite::cv::face::align::PFLD68* |   pytorch_face_landmarks_pfld.onnx  | [...landmark](https://github.com/cunjian/pytorch_face_landmark) | 2.8Mb |
| *lite::cv::face::align::FaceLandmarks1000* |   FaceLandmark1000.onnx  | [FaceLandm...](https://github.com/Single430/FaceLandmark1000) | 2.0Mb |


## Face Attributes.  

<div id="lite.ai.hub.onnx-face-attributes"></div>  


|                  Class                  |          Pretrained ONNX Files           |             Rename or Converted From (Repo)              | Size  |
| :-------------------------------------: | :--------------------------------------: | :------------------------------------------------------: | :---: |
|  *lite::cv::face::attr::AgeGoogleNet*   |            age_googlenet.onnx            |      [onnx-models](https://github.com/onnx/models)       | 23Mb  |
| *lite::cv::face::attr::GenderGoogleNet* |          gender_googlenet.onnx           |      [onnx-models](https://github.com/onnx/models)       | 23Mb  |
| *lite::cv::face::attr::EmotionFerPlus*  |          emotion-ferplus-7.onnx          |      [onnx-models](https://github.com/onnx/models)       | 33Mb  |
| *lite::cv::face::attr::EmotionFerPlus*  |          emotion-ferplus-8.onnx          |      [onnx-models](https://github.com/onnx/models)       | 33Mb  |
|    *lite::cv::face::attr::VGG16Age*     |     vgg_ilsvrc_16_age_imdb_wiki.onnx     |      [onnx-models](https://github.com/onnx/models)       | 514Mb |
|    *lite::cv::face::attr::VGG16Age*     | vgg_ilsvrc_16_age_chalearn_iccv2015.onnx |      [onnx-models](https://github.com/onnx/models)       | 514Mb |
|   *lite::cv::face::attr::VGG16Gender*   |   vgg_ilsvrc_16_gender_imdb_wiki.onnx    |      [onnx-models](https://github.com/onnx/models)       | 512Mb |
|     *lite::cv::face::attr::SSRNet*      |               ssrnet.onnx                | [SSR_Net...](https://github.com/oukohou/SSR_Net_Pytorch) | 190Kb |
|     *lite::cv::face::attr::EfficientEmotion7*      |  face-emotion-recognition-enet_b0_7.onnx  | [face-emo...](https://github.com/HSE-asavchenko/face-emotion-recognition) | 15Mb |
|     *lite::cv::face::attr::EfficientEmotion8*      |  face-emotion-recognition-enet_b0_8_best_afew.onnx | [face-emo...](https://github.com/HSE-asavchenko/face-emotion-recognition) | 15Mb |
|     *lite::cv::face::attr::EfficientEmotion8*      |  face-emotion-recognition-enet_b0_8_best_vgaf.onnx | [face-emo...](https://github.com/HSE-asavchenko/face-emotion-recognition) | 15Mb |
|     *lite::cv::face::attr::MobileEmotion7*      |   face-emotion-recognition-mobilenet_7.onnx  | [face-emo...](https://github.com/HSE-asavchenko/face-emotion-recognition)| 13Mb |
|     *lite::cv::face::attr::ReXNetEmotion7*      | face-emotion-recognition-affectnet_7_vggface2_rexnet150.onnx | [face-emo...](https://github.com/HSE-asavchenko/face-emotion-recognition) | 30Mb |


## Face Recognition.  

<div id="lite.ai.hub.onnx-face-recognition"></div>  


|                   Class                   |            Pretrained ONNX Files             |               Rename or Converted From (Repo)                | Size  |
| :---------------------------------------: | :------------------------------------------: | :----------------------------------------------------------: | :---: |
|     *lite::cv::faceid::GlintArcFace*      |           ms1mv3_arcface_r100.onnx           |  [insightface](https://github.com/deepinsight/insightface)   | 248Mb |
|     *lite::cv::faceid::GlintArcFace*      |           ms1mv3_arcface_r50.onnx            |  [insightface](https://github.com/deepinsight/insightface)   | 166Mb |
|     *lite::cv::faceid::GlintArcFace*      |           ms1mv3_arcface_r34.onnx            |  [insightface](https://github.com/deepinsight/insightface)   | 130Mb |
|     *lite::cv::faceid::GlintArcFace*      |           ms1mv3_arcface_r18.onnx            |  [insightface](https://github.com/deepinsight/insightface)   | 91Mb  |
|     *lite::cv::faceid::GlintCosFace*      |         glint360k_cosface_r100.onnx          |  [insightface](https://github.com/deepinsight/insightface)   | 248Mb |
|     *lite::cv::faceid::GlintCosFace*      |          glint360k_cosface_r50.onnx          |  [insightface](https://github.com/deepinsight/insightface)   | 166Mb |
|     *lite::cv::faceid::GlintCosFace*      |          glint360k_cosface_r34.onnx          |  [insightface](https://github.com/deepinsight/insightface)   | 130Mb |
|     *lite::cv::faceid::GlintCosFace*      |          glint360k_cosface_r18.onnx          |  [insightface](https://github.com/deepinsight/insightface)   | 91Mb  |
|    *lite::cv::faceid::GlintPartialFC*     |        partial_fc_glint360k_r100.onnx        |  [insightface](https://github.com/deepinsight/insightface)   | 248Mb |
|    *lite::cv::faceid::GlintPartialFC*     |        partial_fc_glint360k_r50.onnx         |  [insightface](https://github.com/deepinsight/insightface)   | 91Mb  |
|        *lite::cv::faceid::FaceNet*        |         facenet_vggface2_resnet.onnx         |  [facenet...](https://github.com/timesler/facenet-pytorch)   | 89Mb  |
|        *lite::cv::faceid::FaceNet*        |      facenet_casia-webface_resnet.onnx       |  [facenet...](https://github.com/timesler/facenet-pytorch)   | 89Mb  |
|     *lite::cv::faceid::FocalArcFace*      |        focal-arcface-ms1m-ir152.onnx         | [face.evoLVe...](https://github.com/ZhaoJ9014/face.evoLVe.PyTorch) | 269Mb |
|     *lite::cv::faceid::FocalArcFace*      |    focal-arcface-ms1m-ir50-epoch120.onnx     | [face.evoLVe...](https://github.com/ZhaoJ9014/face.evoLVe.PyTorch) | 166Mb |
|     *lite::cv::faceid::FocalArcFace*      |     focal-arcface-ms1m-ir50-epoch63.onnx     | [face.evoLVe...](https://github.com/ZhaoJ9014/face.evoLVe.PyTorch) | 166Mb |
|   *lite::cv::faceid::FocalAsiaArcFace*    |       focal-arcface-bh-ir50-asia.onnx        | [face.evoLVe...](https://github.com/ZhaoJ9014/face.evoLVe.PyTorch) | 166Mb |
| *lite::cv::faceid::TencentCurricularFace* |     Tencent_CurricularFace_Backbone.onnx     |          [TFace](https://github.com/Tencent/TFace)           | 249Mb |
|    *lite::cv::faceid::TencentCifpFace*    |  Tencent_Cifp_BUPT_Balancedface_IR_34.onnx   |          [TFace](https://github.com/Tencent/TFace)           | 130Mb |
|    *lite::cv::faceid::CenterLossFace*     |        CenterLossFace_epoch_100.onnx         | [center-loss...](https://github.com/louis-she/center-loss.pytorch) | 280Mb |
|      *lite::cv::faceid::SphereFace*       |           sphere20a_20171020.onnx            | [sphere...](https://github.com/clcarwin/sphereface_pytorch)  | 86Mb  |
|    *lite::cv::faceid::PoseRobustFace*     |         dream_cfp_res50_end2end.onnx         |        [DREAM](https://github.com/penincillin/DREAM)         | 92Mb  |
|    *lite::cv::faceid::PoseRobustFace*     |        dream_ijba_res18_end2end.onnx         |        [DREAM](https://github.com/penincillin/DREAM)         | 43Mb  |
|  *lite::cv::faceid:NaivePoseRobustFace*   |          dream_cfp_res50_naive.onnx          |        [DREAM](https://github.com/penincillin/DREAM)         | 91Mb  |
|  *lite::cv::faceid:NaivePoseRobustFace*   |         dream_ijba_res18_naive.onnx          |        [DREAM](https://github.com/penincillin/DREAM)         | 43Mb  |
|     *lite::cv::faceid:MobileFaceNet*      |        MobileFaceNet_Pytorch_068.onnx        | [MobileFace...](https://github.com/Xiaoccer/MobileFaceNet_Pytorch) | 3.8Mb |
|    *lite::cv::faceid:CavaGhostArcFace*    | cavaface_GhostNet_x1.3_Arcface_Epoch_24.onnx | [cavaface...](https://github.com/cavalleria/cavaface.pytorch) | 15Mb  |
|    *lite::cv::faceid:CavaCombinedFace*    |  cavaface_IR_SE_100_Combined_Epoch_24.onnx   | [cavaface...](https://github.com/cavalleria/cavaface.pytorch) | 250Mb |
|    *lite::cv::faceid:MobileSEFocalFace*   | face_recognition.pytorch_Mobilenet_se_focal_121000.onnx | [face_recog...](https://github.com/grib0ed0v/face_recognition.pytorch) | 4.5Mb |


## Head Pose Estimation.  

<div id="lite.ai.hub.onnx-head-pose-estimation"></div>  


|             Class              | Pretrained ONNX Files |               Rename or Converted From (Repo)                | Size  |
| :----------------------------: | :-------------------: | :----------------------------------------------------------: | :---: |
| *lite::cv::face::pose::FSANet* |    fsanet-var.onnx    | [...fsanet...](https://github.com/omasaht/headpose-fsanet-pytorch) | 1.2Mb |
| *lite::cv::face::pose::FSANet* |    fsanet-1x1.onnx    | [...fsanet...](https://github.com/omasaht/headpose-fsanet-pytorch) | 1.2Mb |


## Segmentation.  

<div id="lite.ai.hub.onnx-segmentation"></div>  


|                    Class                     |     Pretrained ONNX Files     |         Rename or Converted From (Repo)          | Size  |
| :------------------------------------------: | :---------------------------: | :----------------------------------------------: | :---: |
| *lite::cv::segmentation::DeepLabV3ResNet101* | deeplabv3_resnet101_coco.onnx | [torchvision](https://github.com/pytorch/vision) | 232Mb |
|    *lite::cv::segmentation::FCNResNet101*    |      fcn_resnet101.onnx       | [torchvision](https://github.com/pytorch/vision) | 207Mb |


## Style Transfer.  

<div id="lite.ai.hub.onnx-style-transfer"></div>  

|                Class                 |   Pretrained ONNX Files    |        Rename or Converted From (Repo)        | Size  |
| :----------------------------------: | :------------------------: | :-------------------------------------------: | :---: |
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


## Colorization.  

<div id="lite.ai.hub.onnx-colorization"></div>

|                Class                |   Pretrained ONNX Files   |              Rename or Converted From (Repo)              | Size  |
| :---------------------------------: | :-----------------------: | :-------------------------------------------------------: | :---: |
| *lite::cv::colorization::Colorizer* |   eccv16-colorizer.onnx   | [colorization](https://github.com/richzhang/colorization) | 123Mb |
| *lite::cv::colorization::Colorizer* | siggraph17-colorizer.onnx | [colorization](https://github.com/richzhang/colorization) | 129Mb |


## Super Resolution.  

<div id="lite.ai.hub.onnx-super-resolution"></div>

|                Class                | Pretrained ONNX Files |              Rename or Converted From (Repo)              | Size  |
| :---------------------------------: | :-------------------: | :-------------------------------------------------------: | :---: |
| *lite::cv::resolution::SubPixelCNN* |   subpixel-cnn.onnx   | [...PIXEL...](https://github.com/niazwazir/SUB_PIXEL_CNN) | 234Kb |

