//
// Created by DefTruth on 2021/3/14.
//

#ifndef LITE_AI_LITE_H
#define LITE_AI_LITE_H

#include "backend.h"

// ENABLE_ONNXRUNTIME
#ifdef ENABLE_ONNXRUNTIME

#include "ort/core/ort_core.h"
#include "ort/core/ort_utils.h"
#include "ort/cv/age_googlenet.h"
#include "ort/cv/glint_arcface.h"
#include "ort/cv/colorizer.h"
#include "ort/cv/deeplabv3_resnet101.h"
#include "ort/cv/densenet.h"
#include "ort/cv/efficientnet_lite4.h"
#include "ort/cv/emotion_ferplus.h"
#include "ort/cv/fast_style_transfer.h"
#include "ort/cv/fcn_resnet101.h"
#include "ort/cv/fsanet.h"
#include "ort/cv/gender_googlenet.h"
#include "ort/cv/ghostnet.h"
#include "ort/cv/hardnet.h"
#include "ort/cv/ibnnet.h"
#include "ort/cv/mobilenetv2.h"
#include "ort/cv/pfld.h"
#include "ort/cv/resnet.h"
#include "ort/cv/resnext.h"
#include "ort/cv/shufflenetv2.h"
#include "ort/cv/ssd.h"
#include "ort/cv/ssd_mobilenetv1.h"
#include "ort/cv/ssrnet.h"
#include "ort/cv/subpixel_cnn.h"
#include "ort/cv/tiny_yolov3.h"
#include "ort/cv/ultraface.h"
#include "ort/cv/vgg16_age.h"
#include "ort/cv/vgg16_gender.h"
#include "ort/cv/yolov3.h"
#include "ort/cv/yolov4.h"
#include "ort/cv/yolov5.h"
#include "ort/cv/glint_cosface.h"
#include "ort/cv/glint_partial_fc.h"
#include "ort/cv/facenet.h"
#include "ort/cv/focal_arcface.h"
#include "ort/cv/focal_asia_arcface.h"
#include "ort/cv/tencent_cifp_face.h"
#include "ort/cv/tencent_curricular_face.h"
#include "ort/cv/center_loss_face.h"
#include "ort/cv/sphere_face.h"

#endif

// ENABLE_MNN
#ifdef ENABLE_MNN
#endif

// ENABLE_NCNN
#ifdef ENABLE_NCNN
#endif

// Default Engine ONNXRuntime
namespace lite
{
  namespace cv
  {
#ifdef BACKEND_ONNXRUNTIME
    namespace utils = ortcv::utils;
    namespace types = ortcv::types;
#endif

#ifdef BACKEND_ONNXRUNTIME
    typedef ortcv::FSANet _FSANet;
    typedef ortcv::PFLD _PFLD;
    typedef ortcv::UltraFace _UltraFace;
    typedef ortcv::AgeGoogleNet _AgeGoogleNet;
    typedef ortcv::GenderGoogleNet _GenderGoogleNet;
    typedef ortcv::EmotionFerPlus _EmotionFerPlus;
    typedef ortcv::VGG16Age _VGG16Age;
    typedef ortcv::VGG16Gender _VGG16Gender;
    typedef ortcv::SSRNet _SSRNet;
    typedef ortcv::FastStyleTransfer _FastStyleTransfer;
    typedef ortcv::GlintArcFace _GlintArcFace;
    typedef ortcv::Colorizer _Colorizer;
    typedef ortcv::SubPixelCNN _SubPixelCNN;
    typedef ortcv::YoloV4 _YoloV4;
    typedef ortcv::YoloV3 _YoloV3;
    typedef ortcv::YoloV5 _YoloV5;
    typedef ortcv::EfficientNetLite4 _EfficientNetLite4;
    typedef ortcv::ShuffleNetV2 _ShuffleNetV2;
    typedef ortcv::TinyYoloV3 _TinyYoloV3;
    typedef ortcv::SSD _SSD;
    typedef ortcv::SSDMobileNetV1 _SSDMobileNetV1;
    typedef ortcv::DeepLabV3ResNet101 _DeepLabV3ResNet101;
    typedef ortcv::DenseNet _DenseNet;
    typedef ortcv::FCNResNet101 _FCNResNet101;
    typedef ortcv::GhostNet _GhostNet;
    typedef ortcv::HdrDNet _HdrDNet;
    typedef ortcv::IBNNet _IBNNet;
    typedef ortcv::MobileNetV2 _MobileNetV2;
    typedef ortcv::ResNet _ResNet;
    typedef ortcv::ResNeXt _ResNeXt;
    typedef ortcv::GlintCosFace _GlintCosFace;
    typedef ortcv::GlintPartialFC _GlintPartialFC;
    typedef ortcv::FaceNet _FaceNet;
    typedef ortcv::FocalArcFace _FocalArcFace;
    typedef ortcv::FocalAsiaArcFace _FocalAsiaArcFace;
    typedef ortcv::TencentCifpFace _TencentCifpFace;
    typedef ortcv::TencentCurricularFace _TencentCurricularFace;
    typedef ortcv::CenterLossFace _CenterLossFace;
    typedef ortcv::SphereFace _SphereFace;
#endif

    // 1. classification
    namespace classification
    {
#ifdef BACKEND_ONNXRUNTIME
      typedef _EfficientNetLite4 EfficientNetLite4;
      typedef _ShuffleNetV2 ShuffleNetV2;
      typedef _DenseNet DenseNet;
      typedef _GhostNet GhostNet;
      typedef _HdrDNet HdrDNet;
      typedef _IBNNet IBNNet;
      typedef _MobileNetV2 MobileNetV2;
      typedef _ResNet ResNet;
      typedef _ResNeXt ResNeXt;
#endif
    }

    // 2. general object detection
    namespace detection
    {
#ifdef BACKEND_ONNXRUNTIME
      typedef _YoloV3 YoloV3;
      typedef _YoloV4 YoloV4;
      typedef _YoloV5 YoloV5;
      typedef _TinyYoloV3 TinyYoloV3;
      typedef _SSD SSD;
      typedef _SSDMobileNetV1 SSDMobileNetV1;
#endif
    }
    // 3. face detection & facial attributes detection
    namespace face
    {
#ifdef BACKEND_ONNXRUNTIME
      typedef _FSANet FSANet; // head pose estimation.
      typedef _UltraFace UltraFace;  // face detection.
      typedef _PFLD PFLD; // facial landmarks detection.
      typedef _AgeGoogleNet AgeGoogleNet; // age estimation
      typedef _GenderGoogleNet GenderGoogleNet; // gender estimation
      typedef _VGG16Age VGG16Age; // age estimation
      typedef _VGG16Gender VGG16Gender; // gender estimation
      typedef _EmotionFerPlus EmotionFerPlus; // emotion detection
      typedef _SSRNet SSRNet; // age estimation
#endif
      namespace detect
      {
#ifdef BACKEND_ONNXRUNTIME
        typedef _UltraFace UltraFace;  // face detection.
#endif
      }

      namespace align
      {
#ifdef BACKEND_ONNXRUNTIME
        typedef _PFLD PFLD; // facial landmarks detection.
#endif
      }

      namespace pose
      {
#ifdef BACKEND_ONNXRUNTIME
        typedef _FSANet FSANet; // head pose estimation.
#endif
      }

      namespace attr
      {
#ifdef BACKEND_ONNXRUNTIME
        typedef _AgeGoogleNet AgeGoogleNet; // age estimation
        typedef _GenderGoogleNet GenderGoogleNet; // gender estimation
        typedef _VGG16Age VGG16Age; // age estimation
        typedef _VGG16Gender VGG16Gender; // gender estimation
        typedef _EmotionFerPlus EmotionFerPlus; // emotion detection
        typedef _SSRNet SSRNet; // age estimation
#endif
      }

    }
    // 4. face recognition
    namespace faceid
    {
#ifdef BACKEND_ONNXRUNTIME
      typedef _GlintArcFace GlintArcFace; //
      typedef _GlintCosFace GlintCosFace; //
      typedef _GlintPartialFC GlintPartialFC;
      typedef _FaceNet FaceNet;
      typedef _FocalArcFace FocalArcFace;
      typedef _FocalAsiaArcFace FocalAsiaArcFace;
      typedef _TencentCurricularFace TencentCurricularFace;
      typedef _TencentCifpFace TencentCifpFace;
      typedef _CenterLossFace CenterLossFace;
      typedef _SphereFace SphereFace;
#endif

    }
    // 5. segmentation
    namespace segmentation
    {
#ifdef BACKEND_ONNXRUNTIME
      typedef _DeepLabV3ResNet101 DeepLabV3ResNet101;
      typedef _FCNResNet101 FCNResNet101;
#endif

    }
    // 6. reid
    namespace reid
    {
#ifdef BACKEND_ONNXRUNTIME
#endif
    }

    // 7. ocr
    namespace ocr
    {
#ifdef BACKEND_ONNXRUNTIME
#endif
    }
    // 8. neural rendering
    namespace render
    {
#ifdef BACKEND_ONNXRUNTIME
#endif
    }
    // 9. style transfer
    namespace style
    {
#ifdef BACKEND_ONNXRUNTIME
      typedef _FastStyleTransfer FastStyleTransfer;
#endif
    }

    // 10. colorization
    namespace colorization
    {
#ifdef BACKEND_ONNXRUNTIME
      typedef _Colorizer Colorizer;
#endif
    }
    // 11. super resolution
    namespace resolution
    {
#ifdef BACKEND_ONNXRUNTIME
      typedef _SubPixelCNN SubPixelCNN;
#endif
    }
    // 12. image & face & human matting
    namespace matting
    {
#ifdef BACKEND_ONNXRUNTIME
#endif
    }

  }

  namespace asr
  {
#ifdef BACKEND_ONNXRUNTIME
#endif
  }

  namespace nlp
  {
#ifdef BACKEND_ONNXRUNTIME
#endif
  }
}

// ONNXRuntime version
namespace lite
{
  namespace onnxruntime
  {
    namespace cv
    {
      namespace utils = ortcv::utils;
      namespace types = ortcv::types;

      typedef ortcv::FSANet _ONNXFSANet;
      typedef ortcv::PFLD _ONNXPFLD;
      typedef ortcv::UltraFace _ONNXUltraFace;
      typedef ortcv::AgeGoogleNet _ONNXAgeGoogleNet;
      typedef ortcv::GenderGoogleNet _ONNXGenderGoogleNet;
      typedef ortcv::EmotionFerPlus _ONNXEmotionFerPlus;
      typedef ortcv::VGG16Age _ONNXVGG16Age;
      typedef ortcv::VGG16Gender _ONNXVGG16Gender;
      typedef ortcv::SSRNet _ONNXSSRNet;
      typedef ortcv::FastStyleTransfer _ONNXFastStyleTransfer;
      typedef ortcv::GlintArcFace _ONNXGlintArcFace;
      typedef ortcv::Colorizer _ONNXColorizer;
      typedef ortcv::SubPixelCNN _ONNXSubPixelCNN;
      typedef ortcv::YoloV4 _ONNXYoloV4;
      typedef ortcv::YoloV3 _ONNXYoloV3;
      typedef ortcv::YoloV5 _ONNXYoloV5;
      typedef ortcv::EfficientNetLite4 _ONNXEfficientNetLite4;
      typedef ortcv::ShuffleNetV2 _ONNXShuffleNetV2;
      typedef ortcv::TinyYoloV3 _ONNXTinyYoloV3;
      typedef ortcv::SSD _ONNXSSD;
      typedef ortcv::SSDMobileNetV1 _ONNXSSDMobileNetV1;
      typedef ortcv::DeepLabV3ResNet101 _ONNXDeepLabV3ResNet101;
      typedef ortcv::DenseNet _ONNXDenseNet;
      typedef ortcv::FCNResNet101 _ONNXFCNResNet101;
      typedef ortcv::GhostNet _ONNXGhostNet;
      typedef ortcv::HdrDNet _ONNXHdrDNet;
      typedef ortcv::IBNNet _ONNXIBNNet;
      typedef ortcv::MobileNetV2 _ONNXMobileNetV2;
      typedef ortcv::ResNet _ONNXResNet;
      typedef ortcv::ResNeXt _ONNXResNeXt;
      typedef ortcv::GlintCosFace _ONNXGlintCosFace;
      typedef ortcv::GlintPartialFC _ONNXGlintPartialFC;
      typedef ortcv::FaceNet _ONNXFaceNet;
      typedef ortcv::FocalArcFace _ONNXFocalArcFace;
      typedef ortcv::FocalAsiaArcFace _ONNXFocalAsiaArcFace;
      typedef ortcv::TencentCifpFace _ONNXTencentCifpFace;
      typedef ortcv::TencentCurricularFace _ONNXTencentCurricularFace;
      typedef ortcv::CenterLossFace _ONNXCenterLossFace;
      typedef ortcv::SphereFace _ONNXSphereFace;

      // 1. classification
      namespace classification
      {
        typedef _ONNXEfficientNetLite4 EfficientNetLite4;
        typedef _ONNXShuffleNetV2 ShuffleNetV2;
        typedef _ONNXDenseNet DenseNet;
        typedef _ONNXGhostNet GhostNet;
        typedef _ONNXHdrDNet HdrDNet;
        typedef _ONNXIBNNet IBNNet;
        typedef _ONNXMobileNetV2 MobileNetV2;
        typedef _ONNXResNet ResNet;
        typedef _ONNXResNeXt ResNeXt;
      }

      // 2. general object detection
      namespace detection
      {
        typedef _ONNXYoloV3 YoloV3;
        typedef _ONNXYoloV4 YoloV4;
        typedef _ONNXYoloV5 YoloV5;
        typedef _ONNXTinyYoloV3 TinyYoloV3;
        typedef _ONNXSSD SSD;
        typedef _ONNXSSDMobileNetV1 SSDMobileNetV1;
      }
      // 3. face detection & facial attributes detection
      namespace face
      {
        typedef _ONNXFSANet FSANet; // head pose estimation.
        typedef _ONNXUltraFace UltraFace;  // face detection.
        typedef _ONNXPFLD PFLD; // facial landmarks detection.
        typedef _ONNXAgeGoogleNet AgeGoogleNet; // age estimation
        typedef _ONNXGenderGoogleNet GenderGoogleNet; // gender estimation
        typedef _ONNXVGG16Age VGG16Age; // age estimation
        typedef _ONNXVGG16Gender VGG16Gender; // gender estimation
        typedef _ONNXEmotionFerPlus EmotionFerPlus; // emotion detection
        typedef _ONNXSSRNet SSRNet; // age estimation

        namespace detect
        {
          typedef _ONNXUltraFace UltraFace;  // face detection.
        }

        namespace align
        {
          typedef _ONNXPFLD PFLD; // facial landmarks detection.
        }

        namespace pose
        {
          typedef _ONNXFSANet FSANet; // head pose estimation.
        }

        namespace attr
        {
          typedef _ONNXAgeGoogleNet AgeGoogleNet; // age estimation
          typedef _ONNXGenderGoogleNet GenderGoogleNet; // gender estimation
          typedef _ONNXVGG16Age VGG16Age; // age estimation
          typedef _ONNXVGG16Gender VGG16Gender; // gender estimation
          typedef _ONNXEmotionFerPlus EmotionFerPlus; // emotion detection
          typedef _ONNXSSRNet SSRNet; // age estimation
        }
      }
      // 4. face recognition
      namespace faceid
      {
        typedef _ONNXGlintArcFace GlintArcFace; //
        typedef _ONNXGlintCosFace GlintCosFace; //
        typedef _ONNXGlintPartialFC GlintPartialFC;
        typedef _ONNXFaceNet FaceNet;
        typedef _ONNXFocalArcFace FocalArcFace;
        typedef _ONNXFocalAsiaArcFace FocalAsiaArcFace;
        typedef _ONNXTencentCifpFace TencentCifpFace;
        typedef _ONNXTencentCurricularFace TencentCurricularFace;
        typedef _ONNXCenterLossFace CenterLossFace;
        typedef _ONNXSphereFace SphereFace;
      }
      // 5. segmentation
      namespace segmentation
      {
        typedef _ONNXDeepLabV3ResNet101 DeepLabV3ResNet101;
        typedef _ONNXFCNResNet101 FCNResNet101;

      }
      // 6. reid
      namespace reid
      {

      }

      // 7. ocr
      namespace ocr
      {

      }
      // 8. neural rendering
      namespace render
      {

      }
      // 9. style transfer
      namespace style
      {
        typedef _ONNXFastStyleTransfer FastStyleTransfer;
      }

      // 10. colorization
      namespace colorization
      {
        typedef _ONNXColorizer Colorizer;
      }
      // 11. super resolution
      namespace resolution
      {
        typedef _ONNXSubPixelCNN SubPixelCNN;
      }
      // 12. image & face & human matting
      namespace matting
      {

      }

    }

    namespace asr
    {

    }

    namespace nlp
    {
    }
  }
}

// MNN version
namespace lite
{
  namespace mnn
  {
  }
}

// NCNN version
namespace lite
{
  namespace ncnn
  {
  }
}

// TNN version
namespace lite
{
  namespace tnn
  {
  }
}
#endif //LITE_AI_LITE_H
