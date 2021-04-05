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
  class SSRNet;              // [8] reference: https://github.com/oukohou/SSR_Net_Pytorch
  class ChineseOCR;          // [9] reference: https://github.com/DayBreak-u/chineseocr_lite
  class ChineseOCRAngleNet;  // [10] reference: https://github.com/DayBreak-u/chineseocr_lite
  class ChineseOCRLiteLSTM;  // [11] reference: https://github.com/DayBreak-u/chineseocr_lite
  class ChineseOCRDBNet;     // [12] reference: https://github.com/DayBreak-u/chineseocr_lite
  class MobileNet;           // [13] reference: https://github.com/onnx/models/blob/master/vision/classification/mobilenet
  class ResNet;              // [14] reference: https://github.com/onnx/models/blob/master/vision/classification/resnet
  class SqueezeNet;          // [15] reference: https://github.com/onnx/models/blob/master/vision/classification/squeezenet
  class ShuffleNetV1;         // [16] reference: https://github.com/onnx/models/blob/master/vision/classification/shufflenet
  class ShuffleNetV2;         // [17] reference: https://github.com/onnx/models/blob/master/vision/classification/shufflenet
  class EfficientNetLite4;    // [18] reference: https://github.com/onnx/models/blob/master/vision/classification/efficientnet-lite4
  class YoloV2;              // [19] reference: https://github.com/onnx/models/blob/master/vision/object_detection_segmentation/yolov2-coco
  class YoloV3;              // [20] reference: https://github.com/onnx/models/blob/master/vision/object_detection_segmentation/yolov3
  class YoloV4;              // [21] reference: https://github.com/onnx/models/blob/master/vision/object_detection_segmentation/yolov4
  class YoloV5;              // [22] reference: https://github.com/ultralytics/yolov5
  class TinyYoloV2;          // [23] reference: https://github.com/onnx/models/blob/master/vision/object_detection_segmentation/tiny-yolov2
  class TinyYoloV3;          // [24] reference: https://github.com/onnx/models/blob/master/vision/object_detection_segmentation/tiny-yolov3
  class SSD;                 // [25] reference: https://github.com/onnx/models/blob/master/vision/object_detection_segmentation/ssd
  class SSDMobileNetV1;      // [26] reference: https://github.com/onnx/models/blob/master/vision/object_detection_segmentation/ssd-mobilenetv1
  class ArcFaceResNet;       // [27] * reference: https://github.com/onnx/models/blob/master/vision/body_analysis/arcface
  class FastStyleTransfer;   // [28] * reference: https://github.com/onnx/models/blob/master/vision/style_transfer/fast_neural_style
  class SubPixelCNN;         // [29] reference: https://github.com/onnx/models/blob/master/vision/super_resolution/sub_pixel_cnn_2016
  class SiggraphColor;       // [30] reference: https://github.com/richzhang/colorization-pytorch
}

namespace ortnlp
{
  class TextCNN;
  class ChineseBert;
}

namespace ortcv { using core::BasicOrtHandler; using core::BasicMultiOrtHandler; }
namespace ortnlp { using core::BasicOrtHandler; using core::BasicMultiOrtHandler; }
namespace ortasr { using core::BasicOrtHandler; using core::BasicMultiOrtHandler; }
#endif //LITEHUB_ORT_ORT_CORE_H
