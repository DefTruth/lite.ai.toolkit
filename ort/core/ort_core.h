//
// Created by DefTruth on 2021/3/18.
//

#ifndef LITEHUB_ORT_ORT_CORE_H
#define LITEHUB_ORT_ORT_CORE_H

#include "__ort_core.h"
#include "ort_handler.h"
#include "ort_types.h"

// namespace cv2 = cv;

namespace ortcv {
  class FSANet;              // reference: https://github.com/omasaht/headpose-fsanet-pytorch
  class PFLD;                // reference: https://github.com/Hsintao/pfld_106_face_landmarks
  class SSRNet;              // reference: https://github.com/oukohou/SSR_Net_Pytorch
  class UltraFace;           // reference: https://github.com/Linzaer/Ultra-Light-Fast-Generic-Face-Detector-1MB
  class ChineseOCR;          // reference: https://github.com/DayBreak-u/chineseocr_lite
  class ChineseOCRAngleNet;  // reference: https://github.com/DayBreak-u/chineseocr_lite
  class ChineseOCRLiteLSTM;  // reference: https://github.com/DayBreak-u/chineseocr_lite
  class ChineseOCRDBNet;     // reference: https://github.com/DayBreak-u/chineseocr_lite
  class MobileNet;           // reference: https://github.com/onnx/models/blob/master/vision/classification/mobilenet
  class ResNet;              // reference: https://github.com/onnx/models/blob/master/vision/classification/resnet
  class SqueezeNet;          // reference: https://github.com/onnx/models/blob/master/vision/classification/squeezenet
  class ShuffleNetV1;         // reference: https://github.com/onnx/models/blob/master/vision/classification/shufflenet
  class ShuffleNetV2;         // reference: https://github.com/onnx/models/blob/master/vision/classification/shufflenet
  class EfficientNetLite4;    // reference: https://github.com/onnx/models/blob/master/vision/classification/efficientnet-lite4
  class YoloV2;              // reference: https://github.com/onnx/models/blob/master/vision/object_detection_segmentation/yolov2-coco
  class YoloV3;              // reference: https://github.com/onnx/models/blob/master/vision/object_detection_segmentation/yolov3
  class YoloV4;              // reference: https://github.com/onnx/models/blob/master/vision/object_detection_segmentation/yolov4
  class YoloV5;              // reference: https://github.com/ultralytics/yolov5
  class TinyYoloV2;          // reference: https://github.com/onnx/models/blob/master/vision/object_detection_segmentation/tiny-yolov2
  class TinyYoloV3;          // reference: https://github.com/onnx/models/blob/master/vision/object_detection_segmentation/tiny-yolov3
  class SSD;                 // reference: https://github.com/onnx/models/blob/master/vision/object_detection_segmentation/ssd
  class SSDMobileNetV1;      // reference: https://github.com/onnx/models/blob/master/vision/object_detection_segmentation/ssd-mobilenetv1
  class AgeGoogleNet;        // reference: https://github.com/onnx/models/tree/master/vision/body_analysis/age_gender
  class GenderGoogleNet;     // reference: https://github.com/onnx/models/tree/master/vision/body_analysis/age_gender
  class ArcFaceResNet;       // reference: https://github.com/onnx/models/blob/master/vision/body_analysis/arcface
  class EmotionFerPlus;      // reference: https://github.com/onnx/models/blob/master/vision/body_analysis/emotion_ferplus
  class VGG16Age;            // reference: https://github.com/onnx/models/tree/master/vision/body_analysis/age_gender
  class FastStyleTransfer;   // reference: https://github.com/onnx/models/blob/master/vision/style_transfer/fast_neural_style
  class SubPixel2016;        // reference: https://github.com/onnx/models/blob/master/vision/super_resolution/sub_pixel_cnn_2016
}

namespace ortnlp {
  class TextCNN;
  class ChineseBert;
}

namespace ortcv { using core::BasicOrtHandler; using core::BasicMultiOrtHandler; }
namespace ortnlp { using core::BasicOrtHandler; using core::BasicMultiOrtHandler; }
namespace ortasr { using core::BasicOrtHandler; using core::BasicMultiOrtHandler; }
#endif //LITEHUB_ORT_ORT_CORE_H
