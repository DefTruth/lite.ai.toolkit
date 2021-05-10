# How to convert SubPixelCNN to ONNX and implements with onnxruntime c++
## 1. 前言

这篇文档主要记录将项目[SUB_PIXEL_CNN](https://github.com/niazwazir/SUB_PIXEL_CNN) 的模型转换成onnx模型，并使用onnxruntime c++接口实现推理的过程。

## 2. 转换成ONNX模型

* 依赖库：
  * pytorch 1.8
  * onnx 1.7.0
  * onnxruntime 1.7.0
  * opencv 4.5.1
  * onnx-simplifier 0.3.5

### 2.1 导出静态维度onnx模型

项目提供了`super_resolution.onnx`文件，但是是动态维度模型。我想要一个固定batch_size=1维度的模型。所以需要对[model_epoch_599.pth](https://github.com/niazwazir/SUB_PIXEL_CNN/blob/master/model_epoch_599.pth)重新导出一个onnx文件。具体代码如下：

```python
import torch
from torch import nn
import torch.nn.init as init

if __name__ == "__main__":
    Path = 'model_epoch_599.pth'
    torch_model = torch.load(Path, map_location="cpu")
    torch_model.eval()
    batch_size = 1  # set as a random number initially (it is set as dynamic axes later)
    # Input to the model
    x = torch.randn(batch_size, 1, 224, 224, requires_grad=True).cpu()  # moving the tensor to GPU
    torch_out = torch_model(x)  # storing the model output to compare with the onnx model output

    # Export the model
    torch.onnx.export(torch_model,  # model being run
                      x,  # model input (or a tuple for multiple inputs)
                      "super_resolution_static.onnx",  # where to save the model (can be a file or file-like object)
                      export_params=True,  # store the trained parameter weights inside the model file
                      opset_version=10,  # the ONNX version to export the model to
                      do_constant_folding=True,  # whether to execute constant folding for optimization
                      input_names=['input'],  # the model's input names
                      output_names=['output']  # the model's output names
                      )
    print("export done.")
    # onnx_model = onnx.load("super_resolution_static.onnx")
    # onnx.checker.check_model(onnx_model)
```

## 3. python版本onnxruntime推理接口

原始项目的推理采用了onnxruntime和PIL的接口做数据预处理，但考虑到在c++，我们想尽可能减少依赖库，于是需要将原始推理中的预处理，用opencv改写。原始的推理逻辑为：

```python
mport io
import cv2
import numpy as np
from torch import nn
import torch.onnx
from PIL import Image
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import onnxruntime

def run_pil():
    onnxruntime.set_default_logger_severity(4)

    ort_session = onnxruntime.InferenceSession("super_resolution.onnx")

    img = Image.open("cat_224x224.jpg")

    resize = transforms.Resize([224, 224])
    img = resize(img)

    img_ycbcr = img.convert('YCbCr')
    img_y, img_cb, img_cr = img_ycbcr.split()

    to_tensor = transforms.ToTensor()
    img_y = to_tensor(img_y)  # (1,224,224) float normalize (0,1)  div -> 255.0
    img_y.unsqueeze_(0)  # (1,1,224,224)

    ort_inputs = {ort_session.get_inputs()[0].name: to_numpy(img_y)}  # The input must be a numpy array
    ort_outs = ort_session.run(None, ort_inputs)
    img_out_y = ort_outs[0]

    print(np.uint8((img_out_y[0] * 255.0).clip(0, 255)[0]).size)
    img_out_y = Image.fromarray(np.uint8((img_out_y[0] * 255.0).clip(0, 255)[0]), mode='L')

    print(img_out_y.size)  # (672,672)

    final_img = Image.merge("YCbCr", [img_out_y,
                                      img_cb.resize(img_out_y.size, Image.BICUBIC),
                                      img_cr.resize(img_out_y.size, Image.BICUBIC),
                                      ]).convert("RGB")
    fig = plt.figure(figsize=[10, 5])

    fig.add_subplot(1, 2, 1, title='Original Image')
    plt.imshow(img)

    fig.add_subplot(1, 2, 2, title='Super resolution Image')
    plt.imshow(final_img)

    fig.subplots_adjust(wspace=0.5)
    plt.show()

    final_img.save("cat_superres_with_ort.jpg")

```

用opencv改写后的接口为：

```python
def run_cv2():
    onnxruntime.set_default_logger_severity(4)
    ort_session = onnxruntime.InferenceSession("super_resolution_static.onnx")

    img = cv2.imread("cat_224x224.jpg")
    img = cv2.resize(img, (224, 224))
    img_ycbcr = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
    img_y, img_cr, img_cb = cv2.split(img_ycbcr)

    print(img_y.shape, img_cb.shape, img_cr.shape)

    img_y = img_y.astype(np.float32) / 255.0  # (224,224) float normalize (0,1)  div -> 255.0
    img_y = np.expand_dims(img_y, 0)
    img_y = np.expand_dims(img_y, 0)  # (1,1,224,224)

    ort_inputs = {ort_session.get_inputs()[0].name: img_y}  # The input must be a numpy array
    ort_outs = ort_session.run(None, ort_inputs)
    img_out_y = ort_outs[0]

    img_out_y = np.uint8((img_out_y[0] * 255.0).clip(0, 255)[0])  # （672，672）
    img_cb = cv2.resize(img_cb, img_out_y.shape)
    img_cr = cv2.resize(img_cr, img_out_y.shape)

    final_img = cv2.merge([img_out_y, img_cr, img_cb])  # YCrCb

    final_img = cv2.cvtColor(final_img, cv2.COLOR_YCrCb2BGR)  # BGR

    cv2.imwrite("cat_superres_with_ort_cv2.jpg", final_img)

    final_img = cv2.cvtColor(final_img, cv2.COLOR_BGR2RGB)  # RGB

    fig = plt.figure(figsize=[10, 5])

    fig.add_subplot(1, 2, 1, title='Original Image')
    plt.imshow(img[:, :, ::-1])

    fig.add_subplot(1, 2, 2, title='Super resolution Image')
    plt.imshow(final_img)

    fig.subplots_adjust(wspace=0.5)
    plt.show()
```

整体的推理逻辑是，将RGB或BGR图像转换为`YCrCb`空间，利用小分辨率上的`Y`预测大分辨率上的`Y_bar`。然后将，预测的`Y_bar`和原图经过resize与`Y_bar`维度对齐的`Cr_resize`以及`Cb_resize`合并成新的`YCrCb`图像。这里需要注意的一个细节是，使用PIL时，是将图像转换成`YCbCr`，而使用opencv时，则是将图像转换成`YCrCb`，这并不影响推理的结果，只要选择`Y`通道作为模型输入即可。

## 4. c++版本onnxruntime推理接口

`subpixel_cnn.h`头文件如下：

```c++
//
// Created by DefTruth on 2021/4/5.
//

#ifndef LITEHUB_ORT_CV_SUBPIXEL_CNN_H
#define LITEHUB_ORT_CV_SUBPIXEL_CNN_H

#include "ort/core/ort_core.h"

namespace ortcv
{
  class SubPixelCNN : public BasicOrtHandler
  {
  public:
    explicit SubPixelCNN(const std::string &_onnx_path, unsigned int _num_threads = 1) :
        BasicOrtHandler(_onnx_path, _num_threads)
    {};

    ~SubPixelCNN() override = default;

  private:
    ort::Value transform(const cv::Mat &mat) override;

  public:
    void detect(const cv::Mat &mat, types::SuperResolutionContent &super_resolution_content);
  };
}

#endif //LITEHUB_ORT_CV_SUBPIXEL_CNN_H
```

`subpixel_cnn.cpp`实现逻辑如下：

```c++
//
// Created by DefTruth on 2021/4/5.
//

#include "subpixel_cnn.h"
#include "ort/core/ort_utils.h"

using ortcv::SubPixelCNN;

ort::Value SubPixelCNN::transform(const cv::Mat &mat)
{
  cv::Mat mat_y; // assume that input mat is Y of YCrCb
  mat.convertTo(mat_y, CV_32FC1, 1.0f / 255.0f, 0.f); // (224,224,1) range (0.,1.0)

  return ortcv::utils::transform::create_tensor(
      mat_y, input_node_dims, memory_info_handler,
      input_values_handler, ortcv::utils::transform::CHW
  );
}

void SubPixelCNN::detect(const cv::Mat &mat, types::SuperResolutionContent &super_resolution_content)
{
  if (mat.empty()) return;
  cv::Mat mat_copy = mat.clone();
  cv::resize(mat_copy, mat_copy, cv::Size(input_node_dims.at(3), input_node_dims.at(2))); // (224,224,3)
  cv::Mat mat_ycrcb, mat_y, mat_cr, mat_cb;
  cv::cvtColor(mat_copy, mat_ycrcb, cv::COLOR_BGR2YCrCb);

  // 0. split
  std::vector<cv::Mat> split_mats;
  cv::split(mat_ycrcb, split_mats);
  mat_y = split_mats.at(0); // (224,224,1) uchar CV_8UC1
  mat_cr = split_mats.at(1);
  mat_cb = split_mats.at(2);

  // 1. make input tensor
  ort::Value input_tensor = this->transform(mat_y);
  // 2. inference
  auto output_tensors = ort_session->Run(
      ort::RunOptions{nullptr}, input_node_names.data(),
      &input_tensor, 1, output_node_names.data(), num_outputs
  );
  ort::Value &pred_tensor = output_tensors.at(0); // (1,1,672,672)
  auto pred_dims = output_node_dims.at(0);
  const unsigned int rows = pred_dims.at(2); // H
  const unsigned int cols = pred_dims.at(3); // W

  mat_y.create(rows, cols, CV_8UC1); // release & create

  for (unsigned int i = 0; i < rows; ++i)
  {
    uchar *p = mat_y.ptr<uchar>(i);
    for (unsigned int j = 0; j < cols; ++j)
    {
      p[j] = cv::saturate_cast<uchar>(pred_tensor.At<float>({0, 0, i, j}) * 255.0f);

    } // CHW->HWC
  }

  cv::resize(mat_cr, mat_cr, cv::Size(cols, rows));
  cv::resize(mat_cb, mat_cb, cv::Size(cols, rows));

  std::vector<cv::Mat> out_mats;
  out_mats.push_back(mat_y);
  out_mats.push_back(mat_cr);
  out_mats.push_back(mat_cb);

  // 3. merge
  cv::merge(out_mats, super_resolution_content.mat);
  if (super_resolution_content.mat.empty())
  {
    super_resolution_content.flag = false;
    return;
  }
  cv::cvtColor(super_resolution_content.mat, super_resolution_content.mat, cv::COLOR_YCrCb2BGR);
  super_resolution_content.flag = true;
}
```

## 5. 编译运行onnxruntime c++推理接口

测试`test_ortcv_subpixel_cnn.cpp`的实现如下。你可以从[Model Zoo](https://github.com/DefTruth/litehub/blob/main/README.md) 下载我转换好的模型。

```c++
//
// Created by DefTruth on 2021/4/5.
//

#include <iostream>
#include <vector>

#include "ort/cv/subpixel_cnn.h"
#include "ort/core/ort_utils.h"

static void test_ortcv_subpixel_cnn()
{
  std::string onnx_path = "../../../hub/onnx/cv/subpixel-cnn.onnx";
  std::string test_img_path = "../../../examples/ort/resources/test_ortcv_subpixel_cnn.jpg";
  std::string save_img_path = "../../../logs/test_ortcv_subpixel_cnn.jpg";

  ortcv::SubPixelCNN *subpixel_cnn = new ortcv::SubPixelCNN(onnx_path);

  ortcv::types::SuperResolutionContent super_resolution_content;
  cv::Mat img_bgr = cv::imread(test_img_path);
  subpixel_cnn->detect(img_bgr, super_resolution_content);

  if (super_resolution_content.flag) cv::imwrite(save_img_path, super_resolution_content.mat);

  std::cout << "Super Resolution Done." << std::endl;

  delete subpixel_cnn;
}

int main(__unused int argc, __unused char *argv[])
{
  test_ortcv_subpixel_cnn();
  return 0;
}
```

工程文件`test_ortcv_subpixel_cnn.cmake`如下：

```cmake
# 1. setup 3rd-party dependences
message(">>>> Current project is [ortcv_subpixel_cnn] in : ${CMAKE_CURRENT_SOURCE_DIR}")
include(${CMAKE_SOURCE_DIR}/setup_3rdparty.cmake)

if (APPLE)
    set(CMAKE_MACOSX_RPATH 1)
    set(CMAKE_BUILD_TYPE release)
endif ()

# 2. setup onnxruntime include
include_directories(${ONNXRUNTIMR_INCLUDE_DIR})
link_directories(${ONNXRUNTIMR_LIBRARY_DIR})

# 3. will be include into CMakeLists.txt at examples/ort
set(ORTCV_FSANET_SRCS
        cv/test_ortcv_subpixel_cnn.cpp
        ${LITEHUB_ROOT_DIR}/ort/cv/subpixel_cnn.cpp
        ${LITEHUB_ROOT_DIR}/ort/core/ort_utils.cpp
        ${LITEHUB_ROOT_DIR}/ort/core/ort_handler.cpp
        )

add_executable(ortcv_subpixel_cnn ${ORTCV_FSANET_SRCS})
target_link_libraries(ortcv_subpixel_cnn
        onnxruntime
        opencv_highgui
        opencv_core
        opencv_imgcodecs
        opencv_imgproc)

if (LITEHUB_COPY_BUILD)
    # "set" only valid in the current directory and subdirectory and does not broadcast
    # to parent and sibling directories
    # CMAKE_SOURCE_DIR means the root path of top CMakeLists.txt
    # CMAKE_CURRENT_SOURCE_DIR the current path of current CMakeLists.txt
    set(EXECUTABLE_OUTPUT_PATH ${CMAKE_SOURCE_DIR}/build/liteort/bin)
    message("=================================================================================")
    message("output binary [app: ortcv_subpixel_cnn] to ${EXECUTABLE_OUTPUT_PATH}")
    message("=================================================================================")
endif ()
```

更具体的工程文件信息，请阅读[examples/ort/CMakeLists.txt](https://github.com/DefTruth/litehub/blob/main/examples/ort/CMakeLists.txt) 以及[examples/ort/cv/test_ortcv_subpiexl_cnn.cmake](https://github.com/DefTruth/litehub/blob/main/examples/ort/cv/test_ortcv_subpixel_cnn.cmake) .

