//
// Created by DefTruth on 2021/10/6.
//

#ifndef LITE_AI_TOOLKIT_MNN_CORE_MNN_CONFIG_H
#define LITE_AI_TOOLKIT_MNN_CORE_MNN_CONFIG_H

#include "mnn_defs.h"

#include <cmath>
#include <vector>
#include <cassert>
#include <locale.h>
#include <string>
#include <algorithm>
#include <memory>
#include <unordered_map>
#include <unordered_set>
#include "opencv2/opencv.hpp"

#ifdef ENABLE_MNN
#include "MNN/Interpreter.hpp"
#include "MNN/MNNDefine.h"
#include "MNN/Tensor.hpp"
#include "MNN/ImageProcess.hpp"
#endif

namespace mnncore {}

#endif //LITE_AI_TOOLKIT_MNN_CORE_MNN_CONFIG_H
