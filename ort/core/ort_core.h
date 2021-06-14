//
// Created by DefTruth on 2021/3/18.
//

#ifndef LITEHUB_ORT_ORT_CORE_H
#define LITEHUB_ORT_ORT_CORE_H

#include "__ort_core.h"
#include "ort_handler.h"
#include "ort_types.h"

// namespace cv2 = cv;

namespace ortcv
{
  class FSANet;              // [0] * reference: https://github.com/omasaht/headpose-fsanet-pytorch
  class PFLD;                // [1] * reference: https://github.com/Hsintao/pfld_106_face_landmarks
  class UltraFace;           // [2] * reference: https://github.com/Linzaer/Ultra-Light-Fast-Generic-Face-Detector-1MB
  class AgeGoogleNet;        // [3] * reference: https://github.com/onnx/models/tree/master/vision/body_analysis/age_gender
  class GenderGoogleNet;     // [4] * reference: https://github.com/onnx/models/tree/master/vision/body_analysis/age_gender
  class EmotionFerPlus;      // [5] * reference: https://github.com/onnx/models/blob/master/vision/body_analysis/emotion_ferplus
  class VGG16Age;            // [6] * reference: https://github.com/onnx/models/tree/master/vision/body_analysis/age_gender
  class VGG16Gender;         // [7] * reference: https://github.com/onnx/models/tree/master/vision/body_analysis/age_gender
  class SSRNet;              // [8] * reference: https://github.com/oukohou/SSR_Net_Pytorch
  class FastStyleTransfer;   // [9] * reference: https://github.com/onnx/models/blob/master/vision/style_transfer/fast_neural_style
  class ArcFaceResNet;       // [10] * reference: https://github.com/onnx/models/blob/master/vision/body_analysis/arcface
  class Colorizer;           // [11] * reference: https://github.com/richzhang/colorization
  class SubPixelCNN;         // [12] * reference: https://github.com/niazwazir/SUB_PIXEL_CNN
  class YoloV4;              // [13] * reference: https://github.com/argusswift/YOLOv4-pytorch
  class YoloV5;              // [14] * reference: https://github.com/ultralytics/yolov5
  class YoloV3;              // [15] * reference: https://github.com/onnx/models/blob/master/vision/object_detection_segmentation/yolov3
  class EfficientNetLite4;   // [16] * reference: https://github.com/onnx/models/blob/master/vision/classification/efficientnet-lite4
  class ShuffleNetV2;        // [17] * reference: https://github.com/onnx/models/blob/master/vision/classification/shufflenet
  class TinyYoloV3;          // [18] * reference: https://github.com/onnx/models/blob/master/vision/object_detection_segmentation/tiny-yolov3
  class SSD;                 // [19] * reference: https://github.com/onnx/models/blob/master/vision/object_detection_segmentation/ssd
  class SSDMobileNetV1;      // [20] * reference: https://github.com/onnx/models/blob/master/vision/object_detection_segmentation/ssd-mobilenetv1
  class DeepLabV3ResNet101;  // [21] * reference: https://pytorch.org/hub/pytorch_vision_deeplabv3_resnet101/
  class DenseNet;            // [22] * reference: https://pytorch.org/hub/pytorch_vision_densenet/
  class FCNResNet101;        // [23] * reference: https://pytorch.org/hub/pytorch_vision_fcn_resnet101/
  class GhostNet;            // [24] * reference：https://pytorch.org/hub/pytorch_vision_ghostnet/
  class HdrDNet;             // [25] * reference: https://pytorch.org/hub/pytorch_vision_hardnet/
  class IBNNet;              // [26] * reference: https://pytorch.org/hub/pytorch_vision_ibnnet/
  class MobileNetV2;         // [27] reference: https://pytorch.org/hub/pytorch_vision_mobilenet_v2/
  class ResNet;              // [28] reference: https://pytorch.org/hub/pytorch_vision_resnet/
  class ResNeXt;             // [29] reference: https://pytorch.org/hub/pytorch_vision_resnext/
  class UNet;                // [30] reference: https://github.com/milesial/Pytorch-UNet
}

namespace ortnlp
{
  class TextCNN;
  class ChineseBert;
  class ChineseOCRLite;
}

namespace ortcv { using core::BasicOrtHandler; using core::BasicMultiOrtHandler; }
namespace ortnlp { using core::BasicOrtHandler; using core::BasicMultiOrtHandler; }
namespace ortasr { using core::BasicOrtHandler; using core::BasicMultiOrtHandler; }
#endif //LITEHUB_ORT_ORT_CORE_H
