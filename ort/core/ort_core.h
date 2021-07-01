//
// Created by DefTruth on 2021/3/18.
//

#ifndef LITEHUB_ORT_ORT_CORE_H
#define LITEHUB_ORT_ORT_CORE_H

#include "ort_config.h"
#include "ort_handler.h"
#include "ort_types.h"

namespace ortcv
{
  class LITEHUB_EXPORTS FSANet;              // [0] * reference: https://github.com/omasaht/headpose-fsanet-pytorch
  class LITEHUB_EXPORTS PFLD;                // [1] * reference: https://github.com/Hsintao/pfld_106_face_landmarks
  class LITEHUB_EXPORTS UltraFace;           // [2] * reference: https://github.com/Linzaer/Ultra-Light-Fast-Generic-Face-Detector-1MB
  class LITEHUB_EXPORTS AgeGoogleNet;        // [3] * reference: https://github.com/onnx/models/tree/master/vision/body_analysis/age_gender
  class LITEHUB_EXPORTS GenderGoogleNet;     // [4] * reference: https://github.com/onnx/models/tree/master/vision/body_analysis/age_gender
  class LITEHUB_EXPORTS EmotionFerPlus;      // [5] * reference: https://github.com/onnx/models/blob/master/vision/body_analysis/emotion_ferplus
  class LITEHUB_EXPORTS VGG16Age;            // [6] * reference: https://github.com/onnx/models/tree/master/vision/body_analysis/age_gender
  class LITEHUB_EXPORTS VGG16Gender;         // [7] * reference: https://github.com/onnx/models/tree/master/vision/body_analysis/age_gender
  class LITEHUB_EXPORTS SSRNet;              // [8] * reference: https://github.com/oukohou/SSR_Net_Pytorch
  class LITEHUB_EXPORTS FastStyleTransfer;   // [9] * reference: https://github.com/onnx/models/blob/master/vision/style_transfer/fast_neural_style
  class LITEHUB_EXPORTS ArcFaceResNet;       // [10] * reference: https://github.com/deepinsight/insightface/tree/master/recognition/arcface_torch
  class LITEHUB_EXPORTS Colorizer;           // [11] * reference: https://github.com/richzhang/colorization
  class LITEHUB_EXPORTS SubPixelCNN;         // [12] * reference: https://github.com/niazwazir/SUB_PIXEL_CNN
  class LITEHUB_EXPORTS YoloV4;              // [13] * reference: https://github.com/argusswift/YOLOv4-pytorch
  class LITEHUB_EXPORTS YoloV5;              // [14] * reference: https://github.com/ultralytics/yolov5
  class LITEHUB_EXPORTS YoloV3;              // [15] * reference: https://github.com/onnx/models/blob/master/vision/object_detection_segmentation/yolov3
  class LITEHUB_EXPORTS EfficientNetLite4;   // [16] * reference: https://github.com/onnx/models/blob/master/vision/classification/efficientnet-lite4
  class LITEHUB_EXPORTS ShuffleNetV2;        // [17] * reference: https://github.com/onnx/models/blob/master/vision/classification/shufflenet
  class LITEHUB_EXPORTS TinyYoloV3;          // [18] * reference: https://github.com/onnx/models/blob/master/vision/object_detection_segmentation/tiny-yolov3
  class LITEHUB_EXPORTS SSD;                 // [19] * reference: https://github.com/onnx/models/blob/master/vision/object_detection_segmentation/ssd
  class LITEHUB_EXPORTS SSDMobileNetV1;      // [20] * reference: https://github.com/onnx/models/blob/master/vision/object_detection_segmentation/ssd-mobilenetv1
  class LITEHUB_EXPORTS DeepLabV3ResNet101;  // [21] * reference: https://pytorch.org/hub/pytorch_vision_deeplabv3_resnet101/
  class LITEHUB_EXPORTS DenseNet;            // [22] * reference: https://pytorch.org/hub/pytorch_vision_densenet/
  class LITEHUB_EXPORTS FCNResNet101;        // [23] * reference: https://pytorch.org/hub/pytorch_vision_fcn_resnet101/
  class LITEHUB_EXPORTS GhostNet;            // [24] * reference：https://pytorch.org/hub/pytorch_vision_ghostnet/
  class LITEHUB_EXPORTS HdrDNet;             // [25] * reference: https://pytorch.org/hub/pytorch_vision_hardnet/
  class LITEHUB_EXPORTS IBNNet;              // [26] * reference: https://pytorch.org/hub/pytorch_vision_ibnnet/
  class LITEHUB_EXPORTS MobileNetV2;         // [27] * reference: https://pytorch.org/hub/pytorch_vision_mobilenet_v2/
  class LITEHUB_EXPORTS ResNet;              // [28] * reference: https://pytorch.org/hub/pytorch_vision_resnet/
  class LITEHUB_EXPORTS ResNeXt;             // [29] * reference: https://pytorch.org/hub/pytorch_vision_resnext/
  class LITEHUB_EXPORTS GlintCosFace;        // [30] * reference: https://github.com/deepinsight/insightface/tree/master/recognition/arcface_torch
}

namespace ortnlp
{
  class LITEHUB_EXPORTS TextCNN; // todo
  class LITEHUB_EXPORTS ChineseBert; // todo
  class LITEHUB_EXPORTS ChineseOCRLite; // todo
}

namespace ortcv
{
  using core::BasicOrtHandler;
  using core::BasicMultiOrtHandler;
}
namespace ortnlp
{
  using core::BasicOrtHandler;
  using core::BasicMultiOrtHandler;
}
namespace ortasr
{
  using core::BasicOrtHandler;
  using core::BasicMultiOrtHandler;
}
#endif //LITEHUB_ORT_ORT_CORE_H
