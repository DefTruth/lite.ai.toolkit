//
// Created by DefTruth on 2021/3/14.
//

#ifndef LITEHUB_LITE_H
#define LITEHUB_LITE_H

#include "backend.h"

#ifdef BACKEND_ONNXRUNTIME

#include "ort/core/ort_core.h"
#include "ort/core/ort_utils.h"
#include "ort/cv/age_googlenet.h"
#include "ort/cv/arcafce_resnet.h"
#include "ort/cv/colorizer.h"
#include "ort/cv/emotion_ferplus.h"
#include "ort/cv/fast_style_transfer.h"
#include "ort/cv/gender_googlenet.h"
#include "ort/cv/pfld.h"
#include "ort/cv/ssrnet.h"
#include "ort/cv/subpixel_cnn.h"
#include "ort/cv/tiny_yolov3.h"
#include "ort/cv/ultraface.h"
#include "ort/cv/vgg16_age.h"
#include "ort/cv/vgg16_gender.h"
#include "ort/cv/yolov3.h"
#include "ort/cv/yolov4.h"
#include "ort/cv/yolov5.h"

#endif

namespace lite
{
  // alias cvx, different with cv(opencv)
  namespace cvx
  {
#ifdef BACKEND_ONNXRUNTIME
    typedef ortcv::AgeGoogleNet _AgeGoogleNet;
    typedef ortcv::ArcFaceResNet _ArcFaceResNet;
    typedef ortcv::Colorizer _Colorizer;
    typedef ortcv::EmotionFerPlus _EmotionFerPlus;
    typedef ortcv::FastStyleTransfer _FastStyleTransfer;
    typedef ortcv::GenderGoogleNet _GenderGoogleNet;
    typedef ortcv::PFLD _PFLD;
    typedef ortcv::SSRNet _SSRNet;
    typedef ortcv::SubPixelCNN _SubPixelCNN;
    typedef ortcv::TinyYoloV3 _TinyYoloV3;
    typedef ortcv::UltraFace _UltraFace;
    typedef ortcv::VGG16Age _VGG16Age;
    typedef ortcv::VGG16Gender _VGG16Gender;
    typedef ortcv::YoloV3 _YoloV3;
    typedef ortcv::YoloV4 _YoloV4;
    typedef ortcv::YoloV5 _YoloV5;
#endif
    // object detection & other detect-like models.
    namespace detection
    {
      // general object detection
      namespace general
      {
#ifdef BACKEND_ONNXRUNTIME
        typedef _YoloV3 YoloV3;
        typedef _YoloV4 YoloV4;
        typedef _YoloV5 YoloV5;
        typedef _TinyYoloV3 TinyYoloV3;
#endif
      }
      // face detection & facial attributes detection
      namespace face
      {
#ifdef BACKEND_ONNXRUNTIME
        typedef _UltraFace UltraFace;  // face detection.
        typedef _PFLD PFLD; // facial landmarks detection.
        typedef _AgeGoogleNet AgeGoogleNet;
        typedef _ArcFaceResNet ArcFaceResNet;
        typedef _AgeGoogleNet AgeGoogleNet;
        typedef _GenderGoogleNet GenderGoogleNet;
        typedef _VGG16Age VGG16Age;
        typedef _VGG16Gender VGG16Gender;
        typedef _EmotionFerPlus EmotionFerPlus;
        typedef _SSRNet SSRNet;
#endif
      }
    }
    // classification
    namespace classification
    {

    }
    // segmentation
    namespace segmentation
    {

    }
    // colorization
    namespace colorization
    {
#ifdef BACKEND_ONNXRUNTIME
      typedef _Colorizer Colorizer;
#endif
    }
    // super resolution
    namespace resolution
    {
#ifdef BACKEND_ONNXRUNTIME
      typedef _SubPixelCNN SubPixelCNN;
#endif
    }
    // style transfer
    namespace style
    {
#ifdef BACKEND_ONNXRUNTIME
      typedef _FastStyleTransfer FastStyleTransfer;
#endif
    }

  }

  namespace asr
  {
  }

  namespace nlp
  {
  }
}

#endif //LITEHUB_LITE_H
