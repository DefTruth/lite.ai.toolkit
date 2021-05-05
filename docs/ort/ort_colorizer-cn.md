# How to convert Colorizer to ONNX and implements with onnxruntime c++

## 1. 前言

这篇文档主要记录将项目[colorization](https://github.com/richzhang/colorization) 的模型转换成onnx模型，并使用onnxruntime c++接口实现推理的过程。

## 2. 转换成ONNX模型

* 依赖库：
  * pytorch 1.8
  * onnx 1.7.0
  * onnxruntime 1.7.0
  * opencv 4.5.1
  *  onnx-simplifier 0.3.5

### 2.1 下载模型文件

原来的模型文件获取是通过torch的`model_zoo.load_url`，但在Mac我们可以直接用`wget`进行下载。

```bash
cd your-path-to-colorization
mkdir pretrained
cd pretrained
wget https://colorizers.s3.us-east-2.amazonaws.com/colorization_release_v2-9b330a0b.pth
wget https://colorizers.s3.us-east-2.amazonaws.com/siggraph17-df00044c.pth
```

### 2.2 修改模型加载代码

将`eccv16.py`和`siggraph17.py`中的代码：

```python
def eccv16(pretrained=True):
	model = ECCVGenerator()
	if(pretrained):
		import torch.utils.model_zoo as model_zoo
		model.load_state_dict(model_zoo.load_url('https://colorizers.s3.us-east-2.amazonaws.com/colorization_release_v2-9b330a0b.pth',map_location='cpu',check_hash=True))
	return model

def siggraph17(pretrained=True):
    model = SIGGRAPHGenerator()
    if(pretrained):
        import torch.utils.model_zoo as model_zoo
        model.load_state_dict(model_zoo.load_url('https://colorizers.s3.us-east-2.amazonaws.com/siggraph17-df00044c.pth',map_location='cpu',check_hash=True))
    return model
```

修改成从本地加载训练好的模型：

```python
def eccv16(pretrained=True, map_location="cpu"):
    model = ECCVGenerator()
    if (pretrained):
        import os
        from pathlib import Path
        root_path = Path(__file__).parent.parent
        pretrained_path = os.path.join(root_path, "pretrained/colorization_release_v2-9b330a0b.pth")
        model.load_state_dict(torch.load(pretrained_path, map_location=map_location))
    return model
  
def siggraph17(pretrained=True, map_location="cpu"):
    model = SIGGRAPHGenerator()
    if (pretrained):
        import os
        from pathlib import Path
        root_path = Path(__file__).parent.parent
        pretrained_path = os.path.join(root_path, "pretrained/siggraph17-df00044c.pth")
        model.load_state_dict(torch.load(pretrained_path, map_location=map_location))
    return model
```

### 2.3 导出成onnx模型文件

`export_onnx.py`的代码逻辑如下，采用了onnx-simplifier优化模型结构：

```python
# -*- coding: utf-8 -*-
import torch
import onnx
import onnxruntime as ort
from onnxsim import simplify

from colorizers import eccv16, siggraph17

def convert_static_eccv16_onnx():
    onnx_path = "./pretrained/eccv16_color.onnx"
    sim_onnx_path = "./pretrained/eccv16_color_sim.onnx"
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    inference_model = eccv16(pretrained=True, map_location=device).to(device)
    inference_model = inference_model.eval()

    x = torch.randn(1, 1, 256, 256).cpu()  # moving the tensor to cpu
    torch.onnx.export(inference_model,  # model being run
                      x,  # model input (or a tuple for multiple inputs)
                      onnx_path,  # where to save the model (can be a file or file-like object)
                      export_params=True,  # store the trained parameter weights inside the model file
                      opset_version=11,  # the ONNX version to export the model to
                      do_constant_folding=True,  # whether to execute constant folding for optimization
                      input_names=['input'],  # the model's input names
                      output_names=['out_ab']  # the model's output names
                      )

    print("export eccv16 onnx done.")
    onnx_model = onnx.load(onnx_path)
    onnx.checker.check_model(onnx_model)

    model_simp, check = simplify(onnx_model, check_n=3)
    onnx.save(model_simp, sim_onnx_path)
    print("export eccv16 onnx sim done.")

    # test onnxruntime
    ort_session = ort.InferenceSession(sim_onnx_path)

    x_numpy = x.cpu().numpy()

    out_ab = ort_session.run(['out_ab'], input_feed={"input": x_numpy})

    print("eccv16 out_ab[0].shape: ", out_ab[0].shape)

def convert_static_siggraph17_onnx():
    onnx_path = "./pretrained/siggraph17_color.onnx"
    sim_onnx_path = "./pretrained/siggraph17_color_sim.onnx"
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    inference_model = siggraph17(pretrained=True, map_location=device).to(device)
    inference_model = inference_model.eval()

    x = torch.randn(1, 1, 256, 256).cpu()  # moving the tensor to cpu
    torch.onnx.export(inference_model,  # model being run
                      x,  # model input (or a tuple for multiple inputs)
                      onnx_path,  # where to save the model (can be a file or file-like object)
                      export_params=True,  # store the trained parameter weights inside the model file
                      opset_version=11,  # the ONNX version to export the model to
                      do_constant_folding=True,  # whether to execute constant folding for optimization
                      input_names=['input'],  # the model's input names
                      output_names=['out_ab']  # the model's output names
                      )

    print("export siggraph17 onnx done.")
    onnx_model = onnx.load(onnx_path)
    onnx.checker.check_model(onnx_model)

    model_simp, check = simplify(onnx_model, check_n=3)
    onnx.save(model_simp, sim_onnx_path)
    print("export siggraph17 onnx sim done.")

    # test onnxruntime
    ort_session = ort.InferenceSession(sim_onnx_path)

    x_numpy = x.cpu().numpy()

    out_ab = ort_session.run(['out_ab'], input_feed={"input": x_numpy})

    print("siggraph17 out_ab[0].shape: ", out_ab[0].shape)


if __name__ == "__main__":
    convert_static_eccv16_onnx()
    convert_static_siggraph17_onnx()

    """cmd
    PYTHONPATH=. python3 ./export_onnx.py
    """

```

## 3. python版本onnxruntime推理接口

原始项目的推理采用了torch和PIL的接口做数据预处理，但考虑到在c++，我们想尽可能减少依赖库，于是需要将原始推理中的预处理，用opencv改写。

原始的pytorch推理实现为：

```python
from PIL import Image
import numpy as np
from skimage import color
import torch
import torch.nn.functional as F

def load_img(img_path):
    out_np = np.asarray(Image.open(img_path))
    if (out_np.ndim == 2):
        out_np = np.tile(out_np[:, :, None], 3)
    return out_np

def resize_img(img, HW=(256, 256), resample=3):
    return np.asarray(Image.fromarray(img).resize((HW[1], HW[0]), resample=resample))

def preprocess_img(img_rgb_orig, HW=(256, 256), resample=3):
    # return original size L and resized L as torch Tensors
    img_rgb_rs = resize_img(img_rgb_orig, HW=HW, resample=resample)
    img_lab_orig = color.rgb2lab(img_rgb_orig)
    img_lab_rs = color.rgb2lab(img_rgb_rs)
    img_l_orig = img_lab_orig[:, :, 0]
    img_l_rs = img_lab_rs[:, :, 0]
    tens_orig_l = torch.Tensor(img_l_orig)[None, None, :, :]
    tens_rs_l = torch.Tensor(img_l_rs)[None, None, :, :]
    return (tens_orig_l, tens_rs_l)

def postprocess_tens(tens_orig_l, out_ab, mode='bilinear'):
    # tens_orig_l 	1 x 1 x H_orig x W_orig
    # out_ab 		1 x 2 x H x W
    HW_orig = tens_orig_l.shape[2:]
    HW = out_ab.shape[2:]
    # call resize function if needed
    if (HW_orig[0] != HW[0] or HW_orig[1] != HW[1]):
        out_ab_orig = F.interpolate(out_ab, size=HW_orig, mode=mode)
    else:
        out_ab_orig = out_ab
    out_lab_orig = torch.cat((tens_orig_l, out_ab_orig), dim=1)
    return color.lab2rgb(out_lab_orig.data.cpu().numpy()[0, ...].transpose((1, 2, 0)))

import argparse
import matplotlib.pyplot as plt
from colorizers import *

def run_demo():
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--img_path', type=str, default='imgs/ansel_adams3.jpg')
    parser.add_argument('--use_gpu', action='store_true', help='whether to use GPU')
    parser.add_argument('-o', '--save_prefix', type=str, default='saved',
                        help='will save into this file with {eccv16.png, siggraph17.png} suffixes')
    opt = parser.parse_args()
    # load colorizers
    colorizer_eccv16 = eccv16(pretrained=True).eval()
    colorizer_siggraph17 = siggraph17(pretrained=True).eval()
    if opt.use_gpu:
        colorizer_eccv16.cuda()
        colorizer_siggraph17.cuda()
    # default size to process images is 256x256
    # grab L channel in both original ("orig") and resized ("rs") resolutions
    img = load_img(opt.img_path)
    (tens_l_orig, tens_l_rs) = preprocess_img(img, HW=(256, 256))
    if opt.use_gpu:
        tens_l_rs = tens_l_rs.cuda()
    # colorizer outputs 256x256 ab map
    # resize and concatenate to original L channel
    img_bw = postprocess_tens(tens_l_orig, torch.cat((0 * tens_l_orig, 0 * tens_l_orig), dim=1))
    out_img_eccv16 = postprocess_tens(tens_l_orig, colorizer_eccv16(tens_l_rs).cpu())
    out_img_siggraph17 = postprocess_tens(tens_l_orig, colorizer_siggraph17(tens_l_rs).cpu())
    plt.imsave('%s_eccv16.png' % opt.save_prefix, out_img_eccv16)
    plt.imsave('%s_siggraph17.png' % opt.save_prefix, out_img_siggraph17)
    
    # 省略 ...

if __name__ == "__main__":
    run_demo()
    """
    PYTHONPATH=. python3 ./demo_release.py -i imgs/ansel_adams3.jpg
    """

```

可以看到，其核心逻辑为4步：

* 第一步：获取灰度图，并转换成3通道图像数据；
* 第二步：将RGB图像（实际上是灰度图，只不过扩张成3通道了）转换成Lab空间，`L`通道表示`亮度`；
* 第三步：将L通道作为模型的输入喂给模型，预测ab两个通道；
* 第四步：将预测的ab通道和真实的L通道进行合并，得到预测的Lab图像，并将Lab转换成RGB，得到预测的RGB图像。

上述前后处理逻辑主要由pytorch、PIL和skimage实现，但在c++部署时，我们只使用opencv的c++接口。于是需要使用opencv的接口来完成相同的操作。

* 参考资料
  * [opencv中BGR/RGB转换成Lab](https://blog.csdn.net/Teddygogogo/article/details/84932648)
  * [opencv转Lab空间需要注意的问题](https://blog.csdn.net/zhangping1987/article/details/73699645?utm_medium=distribute.pc_relevant.none-task-blog-baidujs_title-0&spm=1001.2101.3001.4242)

对于opencv需要先将图像数据转换到`[0.,1.]`，这样在使用COLOR_BGR2Lab后，L通道的值才是在理论值`[0.,100.]`间，否则会落在`[0,255]`区间上。

```python
# -*- coding: utf-8 -*-
import cv2
import numpy as np
import onnxruntime as ort

def infer_eccv16_onnx():
    ort.set_default_logger_severity(4)
    onnx_path = "./pretrained/eccv16_color_sim.onnx"
    ort_session = ort.InferenceSession(onnx_path)
    test_path = "imgs/ansel_adams3.jpg"
    save_path = "imgs_out/saved_eccv16_onnx_1.png"
    img_bgr = cv2.imread(test_path)
    print(img_bgr.shape)
    if img_bgr.ndim == 2:
        img_bgr = np.tile(img_bgr[:, :, None], 3)
    img_bgr_orig = img_bgr  # uint8
    img_bgr_rs = cv2.resize(img_bgr_orig, (256, 256))
    # 对于opencv需要先将图像数据转换到[0.,1.]，这样在使用COLOR_BGR2Lab后，
    # L通道的值才是在理论值[0.,100.]间，否则会落在[0,255]区间上
    img_bgr_rs_norm = img_bgr_rs.astype(np.float32) / 255.0  # (0., 1.)
    img_bgr_orig_norm = img_bgr_orig.astype(np.float32) / 255.0  # (0., 1.)
		# 转换成Lab空间，标准的Lab空间，L的范围为[0.,100.]
    img_lab_orig = cv2.cvtColor(img_bgr_orig_norm, cv2.COLOR_BGR2Lab)
    img_lab_rs = cv2.cvtColor(img_bgr_rs_norm, cv2.COLOR_BGR2Lab)
		# 获取L通道数据
    img_l_orig = img_lab_orig[:, :, 0]  # (?,?)
    img_l_rs = img_lab_rs[:, :, 0]  # (256,256)
    print(np.min(img_l_rs), np.max(img_l_rs))
    img_l_rs = np.expand_dims(img_l_rs, 0)
    img_l_rs = np.expand_dims(img_l_rs, 0)  # (1,1,256,256)
    # 预测ab通道
    out_ab = ort_session.run(['out_ab'], input_feed={"input": img_l_rs})[0]  # (1,2,256,256)
    # 获取原始宽高
    HW_orig = img_l_orig.shape[:]  # H_orig x W_orig
    HW = out_ab.shape[2:]  # 1 x 2 x H x W
    out_ab = out_ab[0].transpose(1, 2, 0)  # HXWX2
    # img_l_orig = np.expand_dims(img_lab_orig, 0) # HXWX1
    # call resize function if needed
    if HW_orig[0] != HW[0] or HW_orig[1] != HW[1]:
        out_ab_orig = cv2.resize(out_ab, (HW_orig[1], HW_orig[0]))
    else:
        out_ab_orig = out_ab
		# 将预测的ab通道与L通道合并
    out_a_orig = out_ab_orig[:, :, 0]
    out_b_orig = out_ab_orig[:, :, 1]
    out_lab_img = cv2.merge([img_l_orig, out_a_orig, out_b_orig])
    # 转换为BGR的uint8图像
    out_bgr_img_norm = cv2.cvtColor(out_lab_img, cv2.COLOR_Lab2BGR)
    out_bgr_img_norm = out_bgr_img_norm * 255.0
    out_bgr_img = out_bgr_img_norm.astype(np.uint8)
    cv2.imwrite(save_path, out_bgr_img)

def infer_siggraph17_onnx():
   ...

if __name__ == "__main__":
    infer_eccv16_onnx()
    infer_siggraph17_onnx()

    """
    PYTHONPATH=. python3 ./inference_onnx.py
    """

```

## 4. c++版本onnxruntime推理接口

`colorizer.h`头文件如下：

```c++
//
// Created by DefTruth on 2021/4/9.
//
#ifndef LITEHUB_ORT_CV_COLORIZER_H
#define LITEHUB_ORT_CV_COLORIZER_H
#include "ort/core/ort_core.h"

namespace ortcv
{
  class Colorizer : public BasicOrtHandler
  {
  public:
    explicit Colorizer(const std::string &_onnx_path, unsigned int _num_threads = 1) :
        BasicOrtHandler(_onnx_path, _num_threads)
    {};

    ~Colorizer() override = default;

  private:
    ort::Value transform(const cv::Mat &mat) override;

  public:
    void detect(const cv::Mat &mat, types::ColorizeContent &colorize_content);
  };
}
#endif //LITEHUB_ORT_CV_COLORIZER_H

```

`colorizer.cpp`实现逻辑如下：

```c++
//
// Created by DefTruth on 2021/4/9.
//

#include "colorizer.h"
#include "ort/core/ort_utils.h"

using ortcv::Colorizer;

ort::Value Colorizer::transform(const cv::Mat &mat)
{
  cv::Mat mat_l; // assume that input mat is L of Lab
  mat.convertTo(mat_l, CV_32FC1, 1.0f, 0.f); // (256,256,1) range (0.,100.)

  return ortcv::utils::transform::create_tensor(
      mat_l, input_node_dims, memory_info_handler,
      input_values_handler, ortcv::utils::transform::CHW
  ); // (1,1,256,256)
}

void Colorizer::detect(const cv::Mat &mat, types::ColorizeContent &colorize_content)
{
  if (mat.empty()) return;
  const unsigned int height = mat.rows;
  const unsigned int width = mat.cols;

  cv::Mat mat_rs = mat.clone();
  cv::resize(mat_rs, mat_rs, cv::Size(input_node_dims.at(3), input_node_dims.at(2))); // (256,256,3)
  cv::Mat mat_rs_norm, mat_orig_norm;
  // 转换为[0.,1.]之间
  mat_rs.convertTo(mat_rs_norm, CV_32FC3, 1.0f / 255.0f, 0.f); // (0.,1.) BGR
  mat.convertTo(mat_orig_norm, CV_32FC3, 1.0f / 255.0f, 0.f); // (0.,1.) BGR
  if (mat_rs_norm.empty() || mat_orig_norm.empty()) return;
  // 将BGR转换的为Lab
  cv::Mat mat_lab_orig, mat_lab_rs;
  cv::cvtColor(mat_rs_norm, mat_lab_rs, cv::COLOR_BGR2Lab);
  cv::cvtColor(mat_orig_norm, mat_lab_orig, cv::COLOR_BGR2Lab);
  // 获取L通道数据
  cv::Mat mat_rs_l, mat_orig_l;
  std::vector<cv::Mat> mats_rs_lab, mats_orig_lab;
  cv::split(mat_lab_rs, mats_rs_lab);
  cv::split(mat_lab_orig, mats_orig_lab);

  mat_rs_l = mats_rs_lab.at(0);
  mat_orig_l = mats_orig_lab.at(0);

  // 1. make input tensor
  ort::Value input_tensor = this->transform(mat_rs_l); // (1,1,256,256)
  auto output_tensors = ort_session->Run(
      ort::RunOptions{nullptr}, input_node_names.data(),
      &input_tensor, 1, output_node_names.data(), num_outputs
  );
  ort::Value &pred_ab_tensor = output_tensors.at(0); // (1,2,256,256)
  auto pred_dims = output_node_dims.at(0);
  const unsigned int rows = pred_dims.at(2); // H 256
  const unsigned int cols = pred_dims.at(3); // W 256
  // 获取预测的ab通道数据
  cv::Mat out_a_orig(rows, cols, CV_32FC1);
  cv::Mat out_b_orig(rows, cols, CV_32FC1);
  // CHW转换为HWC，采用Ort::Value的At接口获取数据
  for (unsigned int i = 0; i < rows; ++i)
  {
    float *pa = out_a_orig.ptr<float>(i);
    float *pb = out_b_orig.ptr<float>(i);
    for (unsigned int j = 0; j < cols; ++j)
    {
      pa[j] = pred_ab_tensor.At<float>({0, 0, i, j});
      pb[j] = pred_ab_tensor.At<float>({0, 1, i, j});
    } // CHW->HWC
  }

  if (rows != height || cols != width)
  {
    cv::resize(out_a_orig, out_a_orig, cv::Size(width, height));
    cv::resize(out_b_orig, out_b_orig, cv::Size(width, height));
  }
	// 合并L、a、b通道并转换为BGR-uint8
  std::vector<cv::Mat> out_mats_lab;
  out_mats_lab.push_back(mat_orig_l);
  out_mats_lab.push_back(out_a_orig);
  out_mats_lab.push_back(out_b_orig);

  cv::Mat merge_mat_lab, mat_bgr_norm;
  cv::merge(out_mats_lab, merge_mat_lab);
  if (merge_mat_lab.empty()) return;
  cv::cvtColor(merge_mat_lab, mat_bgr_norm, cv::COLOR_Lab2BGR); // CV_32FC3
  mat_bgr_norm *= 255.0f;

  mat_bgr_norm.convertTo(colorize_content.mat, CV_8UC3); // uint8

  colorize_content.flag = true;

}
```

## 5. 编译运行onnxruntime c++推理接口

测试`test_ortcv_colorizer.cpp`的实现如下。你可以从[Model Zoo](https://github.com/DefTruth/litehub/blob/main/README.md) 下载我转换好的模型。

```c++
//
// Created by DefTruth on 2021/4/9.
//

#include <iostream>
#include <vector>

#include "ort/cv/colorizer.h"
#include "ort/core/ort_utils.h"

static void test_ortcv_colorizer()
{
  std::string eccv16_onnx_path = "../../../hub/onnx/cv/eccv16-colorizer.onnx";
  std::string siggraph17_onnx_path = "../../../hub/onnx/cv/siggraph17-colorizer.onnx";
  std::string test_img_path1 = "../../../examples/ort/resources/test_ortcv_colorizer_1.jpg";
  std::string test_img_path2 = "../../../examples/ort/resources/test_ortcv_colorizer_2.jpg";
  std::string test_img_path3 = "../../../examples/ort/resources/test_ortcv_colorizer_3.jpg";
  std::string test_img_path4 = "../../../examples/ort/resources/test_ortcv_colorizer_one_piece_0.png";
  std::string save_eccv_img_path1 = "../../../logs/test_ortcv_eccv16_colorizer_1.jpg";
  std::string save_eccv_img_path2 = "../../../logs/test_ortcv_eccv16_colorizer_2.jpg";
  std::string save_eccv_img_path3 = "../../../logs/test_ortcv_eccv16_colorizer_3.jpg";
  std::string save_eccv_img_path4 = "../../../logs/test_ortcv_eccv16_colorizer_one_piece_0.jpg";
  std::string save_siggraph_img_path1 = "../../../logs/test_ortcv_siggraph17_colorizer_1.jpg";
  std::string save_siggraph_img_path2 = "../../../logs/test_ortcv_siggraph17_colorizer_2.jpg";
  std::string save_siggraph_img_path3 = "../../../logs/test_ortcv_siggraph17_colorizer_3.jpg";
  std::string save_siggraph_img_path4 = "../../../logs/test_ortcv_siggraph17_colorizer_one_piece_0.jpg";

  ortcv::Colorizer *eccv16_colorizer = new ortcv::Colorizer(eccv16_onnx_path);
  ortcv::Colorizer *siggraph17_colorizer = new ortcv::Colorizer(siggraph17_onnx_path);

  cv::Mat img_bgr1 = cv::imread(test_img_path1);
  cv::Mat img_bgr2 = cv::imread(test_img_path2);
  cv::Mat img_bgr3 = cv::imread(test_img_path3);
  cv::Mat img_bgr4 = cv::imread(test_img_path4);

  ortcv::types::ColorizeContent eccv16_colorize_content1;
  ortcv::types::ColorizeContent eccv16_colorize_content2;
  ortcv::types::ColorizeContent eccv16_colorize_content3;
  ortcv::types::ColorizeContent eccv16_colorize_content4;

  ortcv::types::ColorizeContent siggraph17_colorize_content1;
  ortcv::types::ColorizeContent siggraph17_colorize_content2;
  ortcv::types::ColorizeContent siggraph17_colorize_content3;
  ortcv::types::ColorizeContent siggraph17_colorize_content4;

  eccv16_colorizer->detect(img_bgr1, eccv16_colorize_content1);
  eccv16_colorizer->detect(img_bgr2, eccv16_colorize_content2);
  eccv16_colorizer->detect(img_bgr3, eccv16_colorize_content3);
  eccv16_colorizer->detect(img_bgr4, eccv16_colorize_content4);

  siggraph17_colorizer->detect(img_bgr1, siggraph17_colorize_content1);
  siggraph17_colorizer->detect(img_bgr2, siggraph17_colorize_content2);
  siggraph17_colorizer->detect(img_bgr3, siggraph17_colorize_content3);
  siggraph17_colorizer->detect(img_bgr4, siggraph17_colorize_content4);

  if (eccv16_colorize_content1.flag) cv::imwrite(save_eccv_img_path1, eccv16_colorize_content1.mat);
  if (eccv16_colorize_content2.flag) cv::imwrite(save_eccv_img_path2, eccv16_colorize_content2.mat);
  if (eccv16_colorize_content3.flag) cv::imwrite(save_eccv_img_path3, eccv16_colorize_content3.mat);
  if (eccv16_colorize_content4.flag) cv::imwrite(save_eccv_img_path4, eccv16_colorize_content4.mat);

  if (siggraph17_colorize_content1.flag) cv::imwrite(save_siggraph_img_path1, siggraph17_colorize_content1.mat);
  if (siggraph17_colorize_content2.flag) cv::imwrite(save_siggraph_img_path2, siggraph17_colorize_content2.mat);
  if (siggraph17_colorize_content3.flag) cv::imwrite(save_siggraph_img_path3, siggraph17_colorize_content3.mat);
  if (siggraph17_colorize_content4.flag) cv::imwrite(save_siggraph_img_path4, siggraph17_colorize_content4.mat);

  std::cout << "Colorization Done." << std::endl;

  delete eccv16_colorizer;
  delete siggraph17_colorizer;
}

int main(__unused int argc, __unused char *argv[])
{
  test_ortcv_colorizer();
  return 0;
}
```

工程文件`test_ortcv_colorizer.cmake`如下：

```cmake
# 1. setup 3rd-party dependences
message(">>>> Current project is [ortcv_colorizer] in : ${CMAKE_CURRENT_SOURCE_DIR}")
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
        cv/test_ortcv_colorizer.cpp
        ${LITEHUB_ROOT_DIR}/ort/cv/colorizer.cpp
        ${LITEHUB_ROOT_DIR}/ort/core/ort_utils.cpp
        ${LITEHUB_ROOT_DIR}/ort/core/ort_handler.cpp
        )

add_executable(ortcv_colorizer ${ORTCV_FSANET_SRCS})
target_link_libraries(ortcv_colorizer
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
    message("output binary [app: ortcv_colorizer] to ${EXECUTABLE_OUTPUT_PATH}")
    message("=================================================================================")
endif ()
```

更具体的工程文件信息，请阅读[examples/ort/CMakeLists.txt](https://github.com/DefTruth/litehub/blob/main/examples/ort/CMakeLists.txt) 以及[examples/ort/cv/test_ortcv_colorizer.cmake](https://github.com/DefTruth/litehub/blob/main/examples/ort/cv/test_ortcv_colorizer.cmake) .