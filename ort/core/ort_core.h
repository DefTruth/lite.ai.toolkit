//
// Created by YanJun Qiu on 2021/3/18.
//

#ifndef LITEHUB_ORT_ORT_CORE_H
#define LITEHUB_ORT_ORT_CORE_H

#include <cmath>
#include <vector>
#include <cassert>
#include <string>
#include <iostream>
#include <algorithm>
#include <unordered_map>
#include <unordered_set>
#include "opencv2/opencv.hpp"
#include "onnxruntime/core/session/onnxruntime_cxx_api.h"

#include "ort_types.h"

#define LITEORT_DEBUG 1

namespace ort = Ort;
// namespace cv2 = cv;

namespace ortcv {

  class FSANet;
  class PFLD;
  class SSRNet;
  class UltraFace;
  class ChineseOCR;
  class ChineseOCRAngleNet;
  class ChineseOCRLiteLSTM;
  class ChineseOCRDBNet;
  class YoloV4;
  class YoloV5;
  class AgeGoogleNet;
  class GenderGoogleNet;
  class ArcFaceResNet;
  class EmotionFerPlus;
  class VGG16Age;

  namespace reid {}
  namespace track {}
  namespace faceid {}
}

namespace ortnlp {
  class TextCNN;
  class ChineseBert;
}

namespace ortasr {

}


#endif //LITEHUB_ORT_ORT_CORE_H
