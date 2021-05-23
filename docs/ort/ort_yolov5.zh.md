# How to convert YoloV5 to ONNX and implements with onnxruntime c++

## 1. 前言

这篇文档主要记录将项目[yolov5](https://github.com/ultralytics/yolov5) 的模型转换成onnx模型，并使用onnxruntime c++接口实现推理的过程。

## 2. 依赖库

* pytorch 1.8
* onnx 1.8.0
* onnxruntime 1.7.0
* opencv 4.5.1
* onnx-simplifier 0.3.5

## 3. 解读Yolov5推理代码

在直接运行`export.py`之前，我们最好还是认真读一遍`yolo.py`，以免转换出现警告或错误时，自己一脸茫然。我在转换yolov5为onnx模型时，还是遇到了一些问题的。这里主要讲讲，yolov5的推理逻辑这块。首先，我们先来看看`Detect`类和`Model`类。相关的问题及理解，我直接写在注释里，方便阅读。在yolov5的[issue](https://github.com/ultralytics/yolov5/issues/251#issuecomment-840984893)提到的`TracerWarning`，以及在无法计算GFLOPs的错误，可以在这篇文章中得到解决。

* 关于`Detect`类，解读如下：

```python
class Detect(nn.Module):
    stride = None  # strides computed during build
    onnx_dynamic = False  # ONNX export parameter

    def __init__(self, nc=80, anchors=(), ch=(), inplace=True):  # detection layer
        super(Detect, self).__init__()
        self.nc = nc  # number of classes nc=80
        self.no = nc + 5  # number of outputs per anchor
        self.nl = len(anchors)  # number of detection layers 预测层的数量 每层负责预测不同尺度的框
        self.na = len(anchors[0]) // 2  # number of anchors 每个点上的anchor数量
        self.grid = [torch.zeros(1)] * self.nl  # init grid
        a = torch.tensor(anchors).float().view(self.nl, -1, 2)  # (nl,na,2) 2表示宽高尺度
        # register_buffer: https://blog.csdn.net/weixin_38145317/article/details/104917218
        # 注册入缓冲区 不会被梯度更新 被视为常量 在forward中可以直接使用
        # self.anchors和self.anchor_grid
        self.register_buffer('anchors', a)  # shape(nl,na,2)
        self.register_buffer('anchor_grid', a.clone().view(self.nl, 1, -1, 1, 1, 2))  # shape(nl,1,na,1,1,2)
        self.m = nn.ModuleList(nn.Conv2d(x, self.no * self.na, 1) for x in ch)  # output conv (bs,na*no,ny,nx)
        self.inplace = inplace  # use in-place ops (e.g. slice assignment)

    def forward(self, x):
        # x = x.copy()  # for profiling
        z = []  # inference output
        for i in range(self.nl):
            x[i] = self.m[i](x[i])  # conv (bs,na*no,ny,nx)
            bs, _, ny, nx = x[i].shape
            # x(bs,255,20,20) to x(bs,3,20,20,85=80+5) (bs,na,ny,nx,no=nc+5=4+1+nc)
            x[i] = x[i].view(bs, self.na, self.no, ny, nx).permute(0, 1, 3, 4, 2).contiguous()

            # 关于self.grid的理解：
            # 一般情况下在training模式下是用不到grid的，因此按理说，grid[i]应该一直是torch.zeros(1)
            # 所以原来这个self.grid[i].shape[2:4] != x[i].shape[2:4]判断必然为True；然而需要注意的
            # 是，在train.py中有一段测试逻辑test.test(...)，这里边调用了model.eval()，而这个方法会把
            # 父类nn.Module的属性self.training设置为False，于是在eval模式下的forward会跑入以下这段
            # 逻辑，从修改了self.grid，所以在我们冻结权重时会发现print出来的grid[i].shape不是(1,)；
            # 问题在于修改后的self.grid所对应的图片尺寸以及网格尺寸，不一定就是我们冻结权重时想要的,
            # 所以self.grid[i].shape[2:4] != x[i].shape[2:4]可能为True也可能为False，这就影响了
            # jit的Tracing(TracerWarning:)。所以解决问题的方法就是，去掉这个判断，始终根据目前的输入维度构造新的grid
            # 从逻辑上看，这并没有改变yolov5最终的推理结果
            if not self.training:  # inference
                # if self.grid[i].shape[2:4] != x[i].shape[2:4] or self.onnx_dynamic:
                #     self.grid[i] = self._make_grid(nx, ny).to(x[i].device)

                # update at 20210515 DefTruth
                self.grid[i] = self._make_grid(nx, ny).to(x[i].device)
                # update at 20210515 DefTruth

                y = x[i].sigmoid()  # (bs,na,ny,nx,no=nc+5=4+1+nc)
                # 应该是将预测的偏移量也做了归一化(0.,1.) 于是xywh+conf+cls_prob都是(0.,1.)
                # 或者说 按照下面的反算逻辑 应该预测的是xy相对于grid[i]上锚点中心的偏移 这种偏移
                # 被限制在(0.,1.)之间；比如grid[i]在(ii,jj)位置上的值2值即为(ii,jj)，代表的是
                # 锚点中心的坐标，预测的是相对于(ii,jj)的偏移量 在(0.,1.)之间

                # 另外在转onnx时 计算GFLOPS出现异常 AttributeError: 'Detect' object has no attribute 'inplace'
                # 估计是训练时保存的模型没有self.inplace这个属性，但是代码后来又添加了这段逻辑 注释掉self.inplace
                # 之后就可以计算GFLOPS了
                # update at 20210515 DefTruth
                xy = (y[..., 0:2] * 2. - 0.5 + self.grid[i]) * self.stride[i]  # xy (bs,na,ny,nx,2)
                wh = (y[..., 2:4] * 2) ** 2 * self.anchor_grid[i].view(1, self.na, 1, 1, 2)  # wh (bs,na,ny,nx,2)
                y = torch.cat((xy, wh, y[..., 4:]), -1) # (bs,na,ny,nx,2+2+1+nc=xy+wh+conf+cls_prob)
                # update at 20210515 DefTruth

                # 在转换成onnx时，默认self.inplance=False
                # if self.inplace:
                #     y[..., 0:2] = (y[..., 0:2] * 2. - 0.5 + self.grid[i]) * self.stride[i]  # xy
                #     y[..., 2:4] = (y[..., 2:4] * 2) ** 2 * self.anchor_grid[i]  # wh
                # else:  # for YOLOv5 on AWS Inferentia https://github.com/ultralytics/yolov5/pull/2953
                #     # 计算预测的中心点 并反算到输入图像的尺寸坐标上
                #     xy = (y[..., 0:2] * 2. - 0.5 + self.grid[i]) * self.stride[i]  # xy
                #     wh = (y[..., 2:4] * 2) ** 2 * self.anchor_grid[i].view(1, self.na, 1, 1, 2)  # wh
                #     y = torch.cat((xy, wh, y[..., 4:]), -1)
                z.append(y.view(bs, -1, self.no))  # y (bs,na*ny*nx,no=2+2+1+nc=xy+wh+conf+cls_prob)

        return x if self.training else (torch.cat(z, 1), x)
        # torch.cat(z, 1) (bs,na*ny*nx*nl,no=2+2+1+nc=xy+wh+conf+cls_prob)

    @staticmethod
    def _make_grid(nx=20, ny=20):
        yv, xv = torch.meshgrid([torch.arange(ny), torch.arange(nx)])
        return torch.stack((xv, yv), 2).view((1, 1, ny, nx, 2)).float()
```

* 关于`Model`类，解读如下：

```python
class Model:
  	def __init__(self, ...): 
      ....
      
    def forward_once(self, x, profile=False):
        y, dt = [], []  # outputs
        for m in self.model:
            if m.f != -1:  # if not from previous layer
                # 注释：当m是Detect时，f=[17, 20, 23]，此时为最后的检测层；x更新为从[17, 20, 23]层获取的特征 此时有3个
                x = y[m.f] if isinstance(m.f, int) else [x if j == -1 else y[j] for j in m.f]  # from earlier layers

            if profile:
                o = thop.profile(m, inputs=(x,), verbose=False)[0] / 1E9 * 2 if thop else 0  # FLOPS
                t = time_synchronized()
                for _ in range(10):
                    _ = m(x)
                dt.append((time_synchronized() - t) * 100)
                if m == self.model[0]:
                    logger.info(f"{'time (ms)':>10s} {'GFLOPS':>10s} {'params':>10s}  {'module'}")
                logger.info(f'{dt[-1]:10.2f} {o:10.2f} {m.np:10.0f}  {m.type}')

            # 注释：在最后的Detect层时，输入的x为3个张量，输出为(torch.cat(z, 1), x)
            # 所以展开后是4个输出 第一个是预测结果 其余的是中间层的特征；但是直接用pth推理的
            # 输出长度是2；而转成onnx后，输出是4个结果。估计是onnx将结果展开了。如果你只在onnx
            # 导出时指定一个输出名称output_names=["pred"]，则另外3个的输出的名称会被自动指定，
            # 如： %pred, %778, %876, %974
            x = m(x)  # run
            y.append(x if m.i in self.save else None)  # save output

        if profile:
            logger.info('%.1fms total' % sum(dt))
        return x
```

## 4. 转换成ONNX模型  

由于我只需要转换静态版本onnx，不需要CoreML和TorchScript，所以我们先对`export.py`进行一些修改。并指定onnx输出节点的名称。注意在使用pytorch-1.8版本时，onnx的版本需要使用`>=1.8`的版本，否则转换后的模型会存在`Slice`切片算子错误。当我换成1.8版本的onnx后，问题就消失了。

```python
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # ...
    # parser.add_argument('--include', nargs='+', default=['torchscript', 'onnx', 'coreml'], help='include formats')
    parser.add_argument('--include', nargs='+', default=['onnx'], help='include formats') # 只转换onnx
    # ...
    opt = parser.parse_args()
    # 省略 部分代码
    # ONNX export ------------------------------------------------------------------------------------------------------
    if 'onnx' in opt.include:
        prefix = colorstr('ONNX:')
        try:
            import onnx

            print(f'{prefix} starting export with onnx {onnx.__version__}...')
            f = opt.weights.replace('.pt', '.onnx')  # filename
            # torch.onnx.export(model, img, f, verbose=False, opset_version=12, input_names=['images'],
            #                   dynamic_axes={'images': {0: 'batch', 2: 'height', 3: 'width'},  # size(1,3,640,640)
            #                                 'output': {0: 'batch', 2: 'y', 3: 'x'}} if opt.dynamic else None)
            # 修改为静态节点 并 指定输出节点名称
            torch.onnx.export(model, img, f,
                              verbose=False,
                              opset_version=12,
                              input_names=['images'],
                              output_names=["pred", "output2", "output3", "output4"]
                             )

            # Checks
            model_onnx = onnx.load(f)  # load onnx model
            onnx.checker.check_model(model_onnx)  # check onnx model
            print(onnx.helper.printable_graph(model_onnx.graph))  # print

            # Simplify
            if opt.simplify:
                try:
                    check_requirements(['onnx-simplifier'])
                    import onnxsim

                    print(f'{prefix} simplifying with onnx-simplifier {onnxsim.__version__}...')
                    model_onnx, check = onnxsim.simplify(
                        model_onnx,
                        dynamic_input_shape=opt.dynamic,
                        input_shapes={'images': list(img.shape)} if opt.dynamic else None)
                    assert check, 'assert check failed'
                    onnx.save(model_onnx, f)
                except Exception as e:
                    print(f'{prefix} simplifier failure: {e}')
            print(f'{prefix} export success, saved as {f} ({file_size(f):.1f} MB)')
        except Exception as e:
            print(f'{prefix} export failure: {e}')
    # 省略 部分代码
    # Finish
    print(f'\nExport complete ({time.time() - t:.2f}s). Visualize with https://github.com/lutzroeder/netron.')

```

运行一下命令进行转换：

```shell
python models/export.py --weights yolov5s.pt --img 640 --batch 1 --simplify
```

## 5. NMS模块解读

在使用c++重写推理逻辑之前需要充分理解`pred`的维度含义以及`nms`模块的具体逻辑。相关的解释已经写在了注释里了。

```python
def non_max_suppression(prediction, conf_thres=0.25, iou_thres=0.45, classes=None, agnostic=False, multi_label=False,
                        labels=()):
    """Runs Non-Maximum Suppression (NMS) on inference results
    Returns:
         list of detections, on (n,6) tensor per image [xyxy, conf, cls]
    """
    # prediction (bs,na*ny*nx*nl,no=2+2+1+nc=cxcy+wh+conf+cls_prob)
    nc = prediction.shape[2] - 5  # number of classes
    xc = prediction[..., 4] > conf_thres  # candidates

    # Checks
    assert 0 <= conf_thres <= 1, f'Invalid Confidence threshold {conf_thres}, valid values are between 0.0 and 1.0'
    assert 0 <= iou_thres <= 1, f'Invalid IoU {iou_thres}, valid values are between 0.0 and 1.0'

    # Settings
    min_wh, max_wh = 2, 4096  # (pixels) minimum and maximum box width and height
    max_det = 300  # maximum number of detections per image
    max_nms = 30000  # maximum number of boxes into torchvision.ops.nms()
    time_limit = 10.0  # seconds to quit after
    redundant = True  # require redundant detections
    multi_label &= nc > 1  # multiple labels per box (adds 0.5ms/img)
    merge = False  # use merge-NMS

    t = time.time()
    output = [torch.zeros((0, 6), device=prediction.device)] * prediction.shape[0]
    for xi, x in enumerate(prediction):  # image index, image inference
        # Apply constraints
        # x[((x[..., 2:4] < min_wh) | (x[..., 2:4] > max_wh)).any(1), 4] = 0  # width-height
        x = x[xc[xi]]  # confidence  (?,5+80)

        # Cat apriori labels if autolabelling
        if labels and len(labels[xi]):
            l = labels[xi]
            v = torch.zeros((len(l), nc + 5), device=x.device)
            v[:, :4] = l[:, 1:5]  # box
            v[:, 4] = 1.0  # conf
            v[range(len(l)), l[:, 0].long() + 5] = 1.0  # cls
            x = torch.cat((x, v), 0)

        # If none remain process next image
        if not x.shape[0]:
            continue

        # Compute conf
        x[:, 5:] *= x[:, 4:5]  # conf = obj_conf * cls_conf

        # Box (center x, center y, width, height) to (x1, y1, x2, y2)
        box = xywh2xyxy(x[:, :4])

        # Detections matrix nx6 (xyxy, conf, cls)
        if multi_label:
            i, j = (x[:, 5:] > conf_thres).nonzero(as_tuple=False).T
            x = torch.cat((box[i], x[i, j + 5, None], j[:, None].float()), 1)
        else:  # best class only
            conf, j = x[:, 5:].max(1, keepdim=True)
            x = torch.cat((box, conf, j.float()), 1)[conf.view(-1) > conf_thres]

        # Filter by class
        if classes is not None:
            x = x[(x[:, 5:6] == torch.tensor(classes, device=x.device)).any(1)]

        # Apply finite constraint
        # if not torch.isfinite(x).all():
        #     x = x[torch.isfinite(x).all(1)]

        # Check shape
        n = x.shape[0]  # number of boxes
        if not n:  # no boxes
            continue
        elif n > max_nms:  # excess boxes
            x = x[x[:, 4].argsort(descending=True)[:max_nms]]  # sort by confidence

        # Batched NMS
        c = x[:, 5:6] * (0 if agnostic else max_wh)  # classes
        # 注释：即是将不同类的框 分开进行nms 预测为同一类别的 会采用相同的offset因此
        # 偏移后依然保持聚在一起的相对位置不变；而不同类的因为offset不同，偏移后会距离
        # 更远；这样一些重叠但是不同类别的实体就不会被NMS过滤掉 很巧妙啊~
        boxes, scores = x[:, :4] + c, x[:, 4]  # boxes (offset by class), scores
        i = torchvision.ops.nms(boxes, scores, iou_thres)  # NMS
        if i.shape[0] > max_det:  # limit detections
            i = i[:max_det]
        if merge and (1 < n < 3E3):  # Merge NMS (boxes merged using weighted mean)
            # update boxes as boxes(i,4) = weights(i,n) * boxes(n,4)
            iou = box_iou(boxes[i], boxes) > iou_thres  # iou matrix
            weights = iou * scores[None]  # box weights
            x[i, :4] = torch.mm(weights, x[:, :4]).float() / weights.sum(1, keepdim=True)  # merged boxes
            if redundant:
                i = i[iou.sum(1) > 1]  # require redundancy

        output[xi] = x[i]
        if (time.time() - t) > time_limit:
            print(f'WARNING: NMS time limit {time_limit}s exceeded')
            break  # time limit exceeded

    return output
```

## 6. python版本onnxruntime推理接口

直接贴代码，很好理解，`detect_onnx.py`，这里还是使用了作者的`non_max_suppression`函数。

```python
# -*- coding: utf-8 -*-
import cv2
import torch
import numpy as np
import onnxruntime as ort
from utils.general import non_max_suppression

names = ['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light',
         'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
         'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
         'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard',
         'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
         'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
         'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
         'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear',
         'hair drier', 'toothbrush']

def infer_yolov5s():
    ort.set_default_logger_severity(4)
    onnx_path = "yolov5s.onnx"
    ort_session = ort.InferenceSession(onnx_path)

    outputs_info = ort_session.get_outputs()
    print("num outputs: ", len(outputs_info))
    print(outputs_info)

    test_path = "./data/images/bus.jpg"
    save_path = "./data/images/bus_onnx.jpg"

    img_bgr = cv2.imread(test_path)
    height, width, _ = img_bgr.shape

    img_rgb = img_bgr[:, :, ::-1]
    img_rgb = cv2.resize(img_rgb, (640, 640))
    img = img_rgb.transpose(2, 0, 1).astype(np.float32)  # (3,640,640) RGB

    img /= 255.0

    img = np.expand_dims(img, 0)
    # [1,num_anchors,num_outputs=2+2+1+nc=cxcy+wh+conf+cls_prob]
    pred = ort_session.run(["pred"], input_feed={"images": img})[0]
    pred_tensor = torch.from_numpy(pred).float()

    boxes_tensor = non_max_suppression(pred_tensor)[0]  # [n,6] [x1,y1,x2,y2,conf,cls]

    boxes = boxes_tensor.cpu().numpy().astype(np.float32)

    if boxes.shape[0] == 0:
        print("no bounding boxes detected.")
        return
    scale_w = width / 640.
    scale_h = height / 640.
    boxes[:, 0] *= scale_w
    boxes[:, 1] *= scale_h
    boxes[:, 2] *= scale_w
    boxes[:, 3] *= scale_h

    print(f"detect {boxes.shape[0]} bounding boxes.")

    for i in range(boxes.shape[0]):
        x1, y1, x2, y2, conf, label = boxes[i]
        x1, y1, x2, y2, label = int(x1), int(y1), int(x2), int(y2), int(label)
        cv2.rectangle(img_bgr, (x1, y1), (x2, y2), (0, 255, 0), 2, 2)
        cv2.putText(img_bgr, names[label] + ":{:.2f}".format(conf), (x1, y1),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2, 2)

    cv2.imwrite(save_path, img_bgr)

    print("detect done.")
```

## 7. c++版本onnxruntime推理接口  

然后，需要将python版本的推理接口，使用onnxruntime c++改写。主要需要改写的核心模块有2个：

* 拿到推理输出`pred`后生成bounding boxes，即方法`generate_bboxes`:

  ```c++
  void generate_bboxes(std::vector<types::Boxf> &bbox_collection,
                           std::vector<ort::Value> &output_tensors,
                           float score_threshold, float img_height,
                           float img_width); // rescale & exclude
  ```

* c++版本的nms模块，并且实现python版本中的添加`offset`的技巧（请看关于non_max_suppression的注释），即`offset_nms`：

  ```c++
  void offset_nms(std::vector<types::Boxf> &input, std::vector<types::Boxf> &output, float iou_threshold, unsigned int topk);
  ```

### 7.1 yolov5.h

```c++
//
// Created by DefTruth on 2021/3/14.
//

#ifndef LITEHUB_ORT_CV_YOLOV5_H
#define LITEHUB_ORT_CV_YOLOV5_H

#include "ort/core/ort_core.h"

namespace ortcv
{
  class YoloV5 : public BasicOrtHandler
  {
  public:
    explicit YoloV5(const std::string &_onnx_path, unsigned int _num_threads = 1) :
        BasicOrtHandler(_onnx_path, _num_threads)
    {};

    ~YoloV5() override = default;

  private:
    static constexpr const float mean_val = 0.f;
    static constexpr const float scale_val = 1.0 / 255.f;
    const char *class_names[80] = {
        "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat", "traffic light",
        "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow",
        "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
        "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard",
        "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
        "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch",
        "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote", "keyboard",
        "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase",
        "scissors", "teddy bear", "hair drier", "toothbrush"
    };
    enum NMS
    {
      HARD = 0, BLEND = 1, OFFSET = 2
    };
    static constexpr const unsigned int max_nms = 30000;

  private:
    ort::Value transform(const cv::Mat &mat) override;

    void generate_bboxes(std::vector<types::Boxf> &bbox_collection,
                         std::vector<ort::Value> &output_tensors,
                         float score_threshold, float img_height,
                         float img_width); // rescale & exclude
    void nms(std::vector<types::Boxf> &input, std::vector<types::Boxf> &output,
             float iou_threshold, unsigned int topk, unsigned int nms_type);

  public:
    void detect(const cv::Mat &mat, std::vector<types::Boxf> &detected_boxes,
                float score_threshold = 0.25f, float iou_threshold = 0.45f,
                unsigned int topk = 100, unsigned int nms_type = NMS::OFFSET);

  };
}

#endif //LITEHUB_ORT_CV_YOLOV5_H

```

### 7.2 yolov5.cpp

```c++
//
// Created by DefTruth on 2021/3/14.
//

#include "yolov5.h"
#include "ort/core/ort_utils.h"

using ortcv::YoloV5;

ort::Value YoloV5::transform(const cv::Mat &mat)
{
  cv::Mat canva = mat.clone();
  cv::cvtColor(canva, canva, cv::COLOR_BGR2RGB);
  cv::resize(canva, canva, cv::Size(input_node_dims.at(3),
                                    input_node_dims.at(2)));
  // (1,3,640,640) 1xCXHXW

  ortcv::utils::transform::normalize_inplace(canva, mean_val, scale_val); // float32
  return ortcv::utils::transform::create_tensor(
      canva, input_node_dims, memory_info_handler,
      input_values_handler, ortcv::utils::transform::CHW);
}


void YoloV5::detect(const cv::Mat &mat, std::vector<types::Boxf> &detected_boxes,
                    float score_threshold, float iou_threshold, unsigned int topk,
                    unsigned int nms_type)
{
  if (mat.empty()) return;
  // this->transform(mat);
  float img_height = static_cast<float>(mat.rows);
  float img_width = static_cast<float>(mat.cols);

  // 1. make input tensor
  ort::Value input_tensor = this->transform(mat);
  // 2. inference scores & boxes.
  auto output_tensors = ort_session->Run(
      ort::RunOptions{nullptr}, input_node_names.data(),
      &input_tensor, 1, output_node_names.data(), num_outputs
  );
  // 3. rescale & exclude.
  std::vector<types::Boxf> bbox_collection;
  this->generate_bboxes(bbox_collection, output_tensors, score_threshold, img_height, img_width);
  // 4. hard|blend nms with topk.
  this->nms(bbox_collection, detected_boxes, iou_threshold, topk, nms_type);
}

void YoloV5::generate_bboxes(std::vector<types::Boxf> &bbox_collection,
                             std::vector<ort::Value> &output_tensors,
                             float score_threshold, float img_height,
                             float img_width)
{

  ort::Value &pred = output_tensors.at(0); // (1,n,85=5+80=cxcy+cwch+obj_conf+cls_conf)
  auto pred_dims = output_node_dims.at(0); // (1,n,85)
  const unsigned int num_anchors = pred_dims.at(1); // n = ?
  const unsigned int num_classes = pred_dims.at(2) - 5;
  const float input_height = static_cast<float>(input_node_dims.at(2)); // e.g 640
  const float input_width = static_cast<float>(input_node_dims.at(3)); // e.g 640
  const float scale_height = img_height / input_height;
  const float scale_width = img_width / input_width;

  bbox_collection.clear();
  unsigned int count = 0;
  for (unsigned int i = 0; i < num_anchors; ++i)
  {
    float obj_conf = pred.At<float>({0, i, 4});
    if (obj_conf < score_threshold) continue; // filter first.

    float cls_conf = pred.At<float>({0, i, 5});
    unsigned int label = 0;
    for (unsigned int j = 0; j < num_classes; ++j)
    {
      float tmp_conf = pred.At<float>({0, i, j + 5});
      if (tmp_conf > cls_conf)
      {
        cls_conf = tmp_conf;
        label = j;
      }
    }
    float conf = obj_conf * cls_conf; // cls_conf (0.,1.)
    if (conf < score_threshold) continue; // filter

    float cx = pred.At<float>({0, i, 0});
    float cy = pred.At<float>({0, i, 1});
    float w = pred.At<float>({0, i, 2});
    float h = pred.At<float>({0, i, 3});

    types::Boxf box;
    box.x1 = (cx - w / 2.f) * scale_width;
    box.y1 = (cy - h / 2.f) * scale_height;
    box.x2 = (cx + w / 2.f) * scale_width;
    box.y2 = (cy + h / 2.f) * scale_height;
    box.score = conf;
    box.label = label;
    box.label_text = class_names[label];
    box.flag = true;
    bbox_collection.push_back(box);

    count += 1; // limit boxes for nms.
    if (count > max_nms)
      break;
  }
#if LITEORT_DEBUG
  std::cout << "detected num_anchors: " << num_anchors << "\n";
  std::cout << "generate_bboxes num: " << bbox_collection.size() << "\n";
#endif
}

void YoloV5::nms(std::vector<types::Boxf> &input, std::vector<types::Boxf> &output,
                 float iou_threshold, unsigned int topk, unsigned int nms_type)
{
  if (nms_type == NMS::BLEND) ortcv::utils::blending_nms(input, output, iou_threshold, topk);
  else if (nms_type == NMS::OFFSET) ortcv::utils::offset_nms(input, output, iou_threshold, topk);
  else ortcv::utils::hard_nms(input, output, iou_threshold, topk);
}
```

### 7.3 offset_nms

```c++
// reference: https://github.com/ultralytics/yolov5/blob/master/utils/general.py
void ortcv::utils::offset_nms(std::vector<types::Boxf> &input, std::vector<types::Boxf> &output,
                              float iou_threshold, unsigned int topk)
{
  if (input.empty()) return;
  std::sort(input.begin(), input.end(),
            [](const types::Boxf &a, const types::Boxf &b)
            { return a.score > b.score; });
  const unsigned int box_num = input.size();
  std::vector<int> merged(box_num, 0);

  const float offset = 4096.f;
  /** Add offset according to classes.
   * That is, separate the boxes into categories, and each category performs its
   * own NMS operation. The same offset will be used for those predicted to be of
   * the same category. Therefore, the relative positions of boxes of the same
   * category will remain unchanged. Box of different classes will be farther away
   * after offset, because offsets are different. In this way, some overlapping but
   * different categories of entities are not filtered out by the NMS. Very clever!
   */
  for (unsigned int i = 0; i < box_num; ++i)
  {
    input[i].x1 += static_cast<float>(input[i].label) * offset;
    input[i].y1 += static_cast<float>(input[i].label) * offset;
    input[i].x2 += static_cast<float>(input[i].label) * offset;
    input[i].y2 += static_cast<float>(input[i].label) * offset;
  }

  unsigned int count = 0;
  for (unsigned int i = 0; i < box_num; ++i)
  {
    if (merged[i]) continue;
    std::vector<types::Boxf> buf;

    buf.push_back(input[i]);
    merged[i] = 1;

    for (unsigned int j = i + 1; j < box_num; ++j)
    {
      if (merged[j]) continue;

      float iou = static_cast<float>(input[i].iou_of(input[j]));

      if (iou > iou_threshold)
      {
        merged[j] = 1;
        buf.push_back(input[j]);
      }

    }
    output.push_back(buf[0]);

    // keep top k
    count += 1;
    if (count >= topk)
      break;
  }

  /** Substract offset.*/
  if (!output.empty())
  {
    for (unsigned int i = 0; i < output.size(); ++i)
    {
      output[i].x1 -= static_cast<float>(output[i].label) * offset;
      output[i].y1 -= static_cast<float>(output[i].label) * offset;
      output[i].x2 -= static_cast<float>(output[i].label) * offset;
      output[i].y2 -= static_cast<float>(output[i].label) * offset;
    }
  }

}
```

## 8. 编译运行onnxruntime c++推理接口

测试`test_ortcv_yolov5.cpp`的实现如下。你可以从[Model Zoo](https://github.com/DefTruth/litehub/blob/main/README.md) 下载我转换好的模型。

```c++
//
// Created by DefTruth on 2021/5/18.
//

#pragma clang diagnostic push
#pragma ide diagnostic ignored "UnusedLocalVariable"

#include <iostream>
#include <vector>

#include "ort/cv/yolov5.h"
#include "ort/core/ort_utils.h"


static void test_ortcv_yolov5()
{

  std::string onnx_path = "../../../hub/onnx/cv/yolov5s.onnx";
  std::string test_img_path_1 = "../../../examples/ort/resources/test_ortcv_yolov5_1.jpg";
  std::string test_img_path_2 = "../../../examples/ort/resources/test_ortcv_yolov5_2.jpg";
  std::string save_img_path_1 = "../../../logs/test_ortcv_yolov5_1.jpg";
  std::string save_img_path_2 = "../../../logs/test_ortcv_yolov5_2.jpg";

  ortcv::YoloV5 *yolov5 = new ortcv::YoloV5(onnx_path);

  std::vector<ortcv::types::Boxf> detected_boxes_1;
  cv::Mat img_bgr_1 = cv::imread(test_img_path_1);
  yolov5->detect(img_bgr_1, detected_boxes_1);

  ortcv::utils::draw_boxes_inplace(img_bgr_1, detected_boxes_1);

  cv::imwrite(save_img_path_1, img_bgr_1);

  std::cout << "Detected Boxes Num: " << detected_boxes_1.size() << std::endl;

  std::vector<ortcv::types::Boxf> detected_boxes_2;
  cv::Mat img_bgr_2 = cv::imread(test_img_path_2);
  yolov5->detect(img_bgr_2, detected_boxes_2);

  ortcv::utils::draw_boxes_inplace(img_bgr_2, detected_boxes_2);

  cv::imwrite(save_img_path_2, img_bgr_2);

  std::cout << "Detected Boxes Num: " << detected_boxes_2.size() << std::endl;

  delete yolov5;

}

int main(__unused int argc, __unused char *argv[])
{
  test_ortcv_yolov5();
  return 0;
}

```

工程文件`test_ortcv_yolov5.cmake`如下：

```cmake
# 1. setup 3rd-party dependences
message(">>>> Current project is [ortcv_yolov5] in : ${CMAKE_CURRENT_SOURCE_DIR}")
include(${CMAKE_SOURCE_DIR}/setup_3rdparty.cmake)

if (APPLE)
    set(CMAKE_MACOSX_RPATH 1)
    set(CMAKE_BUILD_TYPE release)
endif ()

# 2. setup onnxruntime include
include_directories(${ONNXRUNTIMR_INCLUDE_DIR})
link_directories(${ONNXRUNTIMR_LIBRARY_DIR})

# 3. will be include into CMakeLists.txt at examples/ort
set(ORTCV_YOLOV5_SRCS
        cv/test_ortcv_yolov5.cpp
        ${LITEHUB_ROOT_DIR}/ort/cv/yolov5.cpp
        ${LITEHUB_ROOT_DIR}/ort/core/ort_utils.cpp
        ${LITEHUB_ROOT_DIR}/ort/core/ort_handler.cpp
        )

add_executable(ortcv_yolov5 ${ORTCV_YOLOV5_SRCS})
target_link_libraries(ortcv_yolov5
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
    message("output binary [app: ortcv_yolov5] to ${EXECUTABLE_OUTPUT_PATH}")
    message("=================================================================================")
endif ()
```

更具体的工程文件信息，请阅读[examples/ort/CMakeLists.txt](https://github.com/DefTruth/litehub/blob/main/examples/ort/CMakeLists.txt) 以及[examples/ort/cv/test_ortcv_yolov5.cmake](https://github.com/DefTruth/litehub/blob/main/examples/ort/cv/test_ortcv_yolov5.cmake) .

