记录YOLOP工程化

## 项目地址

* [YOLOP](https://github.com/hustvl/YOLOP)  
* [DefTruth/YOLOP](https://github.com/DefTruth/YOLOP)

## 转换成ONNX的过程  

为了成功兼容onnx，首先需要对common中的Focus和Detect进行修改。  

* common2.py  
```python
import math
import numpy as np
import torch
import torch.nn as nn
from PIL import Image, ImageDraw


def autopad(k, p=None):  # kernel, padding
    # Pad to 'same'
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]  # auto-pad
    return p


class DepthSeperabelConv2d(nn.Module):
    """
    DepthSeperable Convolution 2d with residual connection
    """

    def __init__(self, inplanes, planes, kernel_size=3, stride=1, downsample=None, act=True):
        super(DepthSeperabelConv2d, self).__init__()
        self.depthwise = nn.Sequential(
            nn.Conv2d(inplanes, inplanes, kernel_size, stride=stride, groups=inplanes, padding=kernel_size // 2,
                      bias=False),
            nn.BatchNorm2d(inplanes)
        )
        # self.depthwise = nn.Conv2d(inplanes, inplanes, kernel_size,
        # stride=stride, groups=inplanes, padding=1, bias=False)
        # self.pointwise = nn.Conv2d(inplanes, planes, 1, bias=False)

        self.pointwise = nn.Sequential(
            nn.Conv2d(inplanes, planes, 1, bias=False),
            nn.BatchNorm2d(planes)
        )
        self.downsample = downsample
        self.stride = stride
        try:
            self.act = nn.Hardswish() if act else nn.Identity()
        except:
            self.act = nn.Identity()

    def forward(self, x):
        # residual = x

        out = self.depthwise(x)
        out = self.act(out)
        out = self.pointwise(out)

        if self.downsample is not None:
            residual = self.downsample(x)
        out = self.act(out)

        return out


class SharpenConv(nn.Module):
    # SharpenConv convolution
    def __init__(self, c1, c2, k=3, s=1, p=None, g=1, act=True):  # ch_in, ch_out, kernel, stride, padding, groups
        super(SharpenConv, self).__init__()
        sobel_kernel = np.array([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]], dtype='float32')
        kenel_weight = np.vstack([sobel_kernel] * c2 * c1).reshape(c2, c1, 3, 3)
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p), groups=g, bias=False)
        self.conv.weight.data = torch.from_numpy(kenel_weight)
        self.conv.weight.requires_grad = False
        self.bn = nn.BatchNorm2d(c2)
        try:
            self.act = nn.Hardswish() if act else nn.Identity()
        except:
            self.act = nn.Identity()

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

    def fuseforward(self, x):
        return self.act(self.conv(x))


class Conv(nn.Module):
    # Standard convolution
    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, act=True):  # ch_in, ch_out, kernel, stride, padding, groups
        super(Conv, self).__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p), groups=g, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        try:
            self.act = nn.Hardswish() if act else nn.Identity()
        except:
            self.act = nn.Identity()

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

    def fuseforward(self, x):
        return self.act(self.conv(x))


class Bottleneck(nn.Module):
    # Standard bottleneck
    def __init__(self, c1, c2, shortcut=True, g=1, e=0.5):  # ch_in, ch_out, shortcut, groups, expansion
        super(Bottleneck, self).__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c_, c2, 3, 1, g=g)
        self.add = shortcut and c1 == c2

    def forward(self, x):
        return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))


class BottleneckCSP(nn.Module):
    # CSP Bottleneck https://github.com/WongKinYiu/CrossStagePartialNetworks
    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):  # ch_in, ch_out, number, shortcut, groups, expansion
        super(BottleneckCSP, self).__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = nn.Conv2d(c1, c_, 1, 1, bias=False)
        self.cv3 = nn.Conv2d(c_, c_, 1, 1, bias=False)
        self.cv4 = Conv(2 * c_, c2, 1, 1)
        self.bn = nn.BatchNorm2d(2 * c_)  # applied to cat(cv2, cv3)
        self.act = nn.LeakyReLU(0.1, inplace=True)
        self.m = nn.Sequential(*[Bottleneck(c_, c_, shortcut, g, e=1.0) for _ in range(n)])

    def forward(self, x):
        y1 = self.cv3(self.m(self.cv1(x)))
        y2 = self.cv2(x)
        return self.cv4(self.act(self.bn(torch.cat((y1, y2), dim=1))))


class SPP(nn.Module):
    # Spatial pyramid pooling layer used in YOLOv3-SPP
    def __init__(self, c1, c2, k=(5, 9, 13)):
        super(SPP, self).__init__()
        c_ = c1 // 2  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c_ * (len(k) + 1), c2, 1, 1)
        self.m = nn.ModuleList([nn.MaxPool2d(kernel_size=x, stride=1, padding=x // 2) for x in k])

    def forward(self, x):
        x = self.cv1(x)
        return self.cv2(torch.cat([x] + [m(x) for m in self.m], 1))


class Contract(nn.Module):
    # Contract width-height into channels, i.e. x(1,64,80,80) to x(1,256,40,40)
    def __init__(self, gain=2):
        super().__init__()
        self.gain = gain

    def forward(self, x):
        N, C, H, W = x.size()  # assert (H / s == 0) and (W / s == 0), 'Indivisible gain'
        s = self.gain
        x = x.view(N, C, H // s, s, W // s, s)  # x(1,64,40,2,40,2)
        x = x.permute(0, 3, 5, 1, 2, 4).contiguous()  # x(1,2,2,64,40,40)
        return x.view(N, C * s * s, H // s, W // s)  # x(1,256,40,40)


class Focus(nn.Module):
    # Focus wh information into c-space
    # slice concat conv
    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, act=True):  # ch_in, ch_out, kernel, stride, padding, groups
        super(Focus, self).__init__()
        self.conv = Conv(c1 * 4, c2, k, s, p, g, act)
        self.contract = Contract(gain=2)

    def forward(self, x):  # x(b,c,w,h) -> y(b,4c,w/2,h/2)
        return self.conv(torch.cat([x[..., ::2, ::2], x[..., 1::2, ::2], x[..., ::2, 1::2], x[..., 1::2, 1::2]], 1))
        # return self.conv(self.contract(x))


class Concat(nn.Module):
    # Concatenate a list of tensors along dimension
    def __init__(self, dimension=1):
        super(Concat, self).__init__()
        self.d = dimension

    def forward(self, x):
        """ print("***********************")
        for f in x:
            print(f.shape) """
        return torch.cat(x, self.d)


class Detect(nn.Module):
    stride = None  # strides computed during build

    def __init__(self, nc=13, anchors=(), ch=()):  # detection layer
        super(Detect, self).__init__()
        self.nc = nc  # number of classes
        self.no = nc + 5  # number of outputs per anchor
        self.nl = len(anchors)  # number of detection layers 3
        self.na = len(anchors[0]) // 2  # number of anchors 3
        self.grid = [torch.zeros(1)] * self.nl  # init grid
        a = torch.tensor(anchors).float().view(self.nl, -1, 2)  # (nl=3,na=3,2)
        self.register_buffer('anchors', a)  # shape(nl,na,2)
        self.register_buffer('anchor_grid', a.clone().view(self.nl, 1, -1, 1, 1, 2))  # shape(nl=3,1,na=3,1,1,2)
        self.m = nn.ModuleList(nn.Conv2d(x, self.no * self.na, 1) for x in ch)  # output conv
        self.inplace = True  # use in-place ops (e.g. slice assignment)

    # def forward(self, x):
    #     z = []  # inference output
    #     for i in range(self.nl):
    #         x[i] = self.m[i](x[i])  # conv
    #         # print(str(i)+str(x[i].shape))
    #         bs, _, ny, nx = x[i].shape  # x(bs,255,w,w) to x(bs,3,w,w,85)
    #         x[i] = x[i].view(bs, self.na, self.no, ny, nx).permute(0, 1, 3, 4, 2).contiguous()
    #         # print(str(i)+str(x[i].shape))
    #
    #         if not self.training:  # inference
    #             if self.grid[i].shape[2:4] != x[i].shape[2:4]:
    #                 self.grid[i] = self._make_grid(nx, ny).to(x[i].device)
    #             y = x[i].sigmoid()
    #             # print("**")
    #             # print(y.shape) #[1, 3, w, h, 85]
    #             # print(self.grid[i].shape) #[1, 3, w, h, 2]
    #             y[..., 0:2] = (y[..., 0:2] * 2. - 0.5 + self.grid[i].to(x[i].device)) * self.stride[i]  # xy
    #             y[..., 2:4] = (y[..., 2:4] * 2) ** 2 * self.anchor_grid[i]  # wh
    #             """print("**")
    #             print(y.shape)  #[1, 3, w, h, 85]
    #             print(y.view(bs, -1, self.no).shape) #[1, 3*w*h, 85]"""
    #             z.append(y.view(bs, -1, self.no))
    #     return x if self.training else (torch.cat(z, 1), x)

    def forward(self, x):
        z = []  # inference output
        for i in range(self.nl):
            x[i] = self.m[i](x[i])  # conv (bs,na*no,ny,nx)
            bs, _, ny, nx = x[i].shape
            # x(bs,255,20,20) to x(bs,3,20,20,nc+5) (bs,na,ny,nx,no=nc+5=4+1+nc)
            x[i] = x[i].view(bs, self.na, self.no, ny, nx).permute(0, 1, 3, 4, 2).contiguous()

            if not self.training:  # inference
                # if self.grid[i].shape[2:4] != x[i].shape[2:4] or self.onnx_dynamic:
                #     self.grid[i] = self._make_grid(nx, ny).to(x[i].device)

                self.grid[i] = self._make_grid(nx, ny).to(x[i].device)

                y = x[i].sigmoid()  # (bs,na,ny,nx,no=nc+5=4+1+nc)

                xy = (y[..., 0:2] * 2. - 0.5 + self.grid[i]) * self.stride[i]  # xy (bs,na,ny,nx,2)
                wh = (y[..., 2:4] * 2) ** 2 * self.anchor_grid[i].view(1, self.na, 1, 1, 2)  # wh (bs,na,ny,nx,2)
                y = torch.cat((xy, wh, y[..., 4:]), -1) # (bs,na,ny,nx,2+2+1+nc=xy+wh+conf+cls_prob)

                z.append(y.view(bs, -1, self.no))  # y (bs,na*ny*nx,no=2+2+1+nc=xy+wh+conf+cls_prob)

        return x if self.training else (torch.cat(z, 1), x)
        # torch.cat(z, 1) (bs,na*ny*nx*nl,no=2+2+1+nc=xy+wh+conf+cls_prob)

    @staticmethod
    def _make_grid(nx=20, ny=20):

        yv, xv = torch.meshgrid([torch.arange(ny), torch.arange(nx)])
        return torch.stack((xv, yv), 2).view((1, 1, ny, nx, 2)).float()


"""class Detections:
    # detections class for YOLOv5 inference results
    def __init__(self, imgs, pred, names=None):
        super(Detections, self).__init__()
        d = pred[0].device  # device
        gn = [torch.tensor([*[im.shape[i] for i in [1, 0, 1, 0]], 1., 1.], device=d) for im in imgs]  # normalizations
        self.imgs = imgs  # list of images as numpy arrays
        self.pred = pred  # list of tensors pred[0] = (xyxy, conf, cls)
        self.names = names  # class names
        self.xyxy = pred  # xyxy pixels
        self.xywh = [xyxy2xywh(x) for x in pred]  # xywh pixels
        self.xyxyn = [x / g for x, g in zip(self.xyxy, gn)]  # xyxy normalized
        self.xywhn = [x / g for x, g in zip(self.xywh, gn)]  # xywh normalized
        self.n = len(self.pred)

    def display(self, pprint=False, show=False, save=False):
        colors = color_list()
        for i, (img, pred) in enumerate(zip(self.imgs, self.pred)):
            str = f'Image {i + 1}/{len(self.pred)}: {img.shape[0]}x{img.shape[1]} '
            if pred is not None:
                for c in pred[:, -1].unique():
                    n = (pred[:, -1] == c).sum()  # detections per class
                    str += f'{n} {self.names[int(c)]}s, '  # add to string
                if show or save:
                    img = Image.fromarray(img.astype(np.uint8)) if isinstance(img, np.ndarray) else img  # from np
                    for *box, conf, cls in pred:  # xyxy, confidence, class
                        # str += '%s %.2f, ' % (names[int(cls)], conf)  # label
                        ImageDraw.Draw(img).rectangle(box, width=4, outline=colors[int(cls) % 10])  # plot
            if save:
                f = f'results{i}.jpg'
                str += f"saved to '{f}'"
                img.save(f)  # save
            if show:
                img.show(f'Image {i}')  # show
            if pprint:
                print(str)

    def print(self):
        self.display(pprint=True)  # print results

    def show(self):
        self.display(show=True)  # show results

    def save(self):
        self.display(save=True)  # save results

    def __len__(self):
        return self.n

    def tolist(self):
        # return a list of Detections objects, i.e. 'for result in results.tolist():'
        x = [Detections([self.imgs[i]], [self.pred[i]], self.names) for i in range(self.n)]
        for d in x:
            for k in ['imgs', 'pred', 'xyxy', 'xyxyn', 'xywh', 'xywhn']:
                setattr(d, k, getattr(d, k)[0])  # pop out of list"""


```  
然后是，修改MCnet，让模型输出想要的3个结果张量。  

* export_onnx.py  
```python
import torch
import torch.nn as nn
from lib.models.common2 import Conv, SPP, Bottleneck, BottleneckCSP, Focus, Concat, Detect, SharpenConv
from torch.nn import Upsample
from lib.utils import check_anchor_order
from lib.utils import initialize_weights
import argparse
import onnx
import onnxruntime as ort
import onnxsim

import math
import cv2

# The lane line and the driving area segment branches without share information with each other and without link
YOLOP = [
    [24, 33, 42],  # Det_out_idx, Da_Segout_idx, LL_Segout_idx
    [-1, Focus, [3, 32, 3]],  # 0
    [-1, Conv, [32, 64, 3, 2]],  # 1
    [-1, BottleneckCSP, [64, 64, 1]],  # 2
    [-1, Conv, [64, 128, 3, 2]],  # 3
    [-1, BottleneckCSP, [128, 128, 3]],  # 4
    [-1, Conv, [128, 256, 3, 2]],  # 5
    [-1, BottleneckCSP, [256, 256, 3]],  # 6
    [-1, Conv, [256, 512, 3, 2]],  # 7
    [-1, SPP, [512, 512, [5, 9, 13]]],  # 8 SPP
    [-1, BottleneckCSP, [512, 512, 1, False]],  # 9
    [-1, Conv, [512, 256, 1, 1]],  # 10
    [-1, Upsample, [None, 2, 'nearest']],  # 11
    [[-1, 6], Concat, [1]],  # 12
    [-1, BottleneckCSP, [512, 256, 1, False]],  # 13
    [-1, Conv, [256, 128, 1, 1]],  # 14
    [-1, Upsample, [None, 2, 'nearest']],  # 15
    [[-1, 4], Concat, [1]],  # 16         #Encoder

    [-1, BottleneckCSP, [256, 128, 1, False]],  # 17
    [-1, Conv, [128, 128, 3, 2]],  # 18
    [[-1, 14], Concat, [1]],  # 19
    [-1, BottleneckCSP, [256, 256, 1, False]],  # 20
    [-1, Conv, [256, 256, 3, 2]],  # 21
    [[-1, 10], Concat, [1]],  # 22
    [-1, BottleneckCSP, [512, 512, 1, False]],  # 23
    [[17, 20, 23], Detect,
     [1, [[3, 9, 5, 11, 4, 20], [7, 18, 6, 39, 12, 31], [19, 50, 38, 81, 68, 157]], [128, 256, 512]]],
    # Detection head 24: from_(features from specific layers), block, nc(num_classes) anchors ch(channels)

    [16, Conv, [256, 128, 3, 1]],  # 25
    [-1, Upsample, [None, 2, 'nearest']],  # 26
    [-1, BottleneckCSP, [128, 64, 1, False]],  # 27
    [-1, Conv, [64, 32, 3, 1]],  # 28
    [-1, Upsample, [None, 2, 'nearest']],  # 29
    [-1, Conv, [32, 16, 3, 1]],  # 30
    [-1, BottleneckCSP, [16, 8, 1, False]],  # 31
    [-1, Upsample, [None, 2, 'nearest']],  # 32
    [-1, Conv, [8, 2, 3, 1]],  # 33 Driving area segmentation head

    [16, Conv, [256, 128, 3, 1]],  # 34
    [-1, Upsample, [None, 2, 'nearest']],  # 35
    [-1, BottleneckCSP, [128, 64, 1, False]],  # 36
    [-1, Conv, [64, 32, 3, 1]],  # 37
    [-1, Upsample, [None, 2, 'nearest']],  # 38
    [-1, Conv, [32, 16, 3, 1]],  # 39
    [-1, BottleneckCSP, [16, 8, 1, False]],  # 40
    [-1, Upsample, [None, 2, 'nearest']],  # 41
    [-1, Conv, [8, 2, 3, 1]]  # 42 Lane line segmentation head
]


class MCnet(nn.Module):
    def __init__(self, block_cfg):
        super(MCnet, self).__init__()
        layers, save = [], []
        self.nc = 1  # traffic or not
        self.detector_index = -1
        self.det_out_idx = block_cfg[0][0]
        self.seg_out_idx = block_cfg[0][1:]
        self.num_anchors = 3
        self.num_outchannel = 5 + self.nc  # dx,dy,dw,dh,obj_conf+cls_conf
        # Build model
        for i, (from_, block, args) in enumerate(block_cfg[1:]):
            block = eval(block) if isinstance(block, str) else block  # eval strings
            if block is Detect:
                self.detector_index = i
            block_ = block(*args)
            block_.index, block_.from_ = i, from_
            layers.append(block_)
            save.extend(x % i for x in ([from_] if isinstance(from_, int) else from_) if x != -1)  # append to savelist
        assert self.detector_index == block_cfg[0][0]

        self.model, self.save = nn.Sequential(*layers), sorted(save)
        self.names = [str(i) for i in range(self.nc)]

        # set stride、anchor for detector
        Detector = self.model[self.detector_index]  # detector
        if isinstance(Detector, Detect):
            s = 128  # 2x min stride
            # for x in self.forward(torch.zeros(1, 3, s, s)):
            #     print (x.shape)
            with torch.no_grad():
                model_out = self.forward(torch.zeros(1, 3, s, s))
                detects, _, _ = model_out
                Detector.stride = torch.tensor([s / x.shape[-2] for x in detects])  # forward
            # print("stride"+str(Detector.stride ))
            Detector.anchors /= Detector.stride.view(-1, 1, 1)  # Set the anchors for the corresponding scale
            check_anchor_order(Detector)
            self.stride = Detector.stride
            # self._initialize_biases()
        initialize_weights(self)

    def forward(self, x):
        cache = []
        out = []
        det_out = None
        for i, block in enumerate(self.model):
            if block.from_ != -1:
                x = cache[block.from_] if isinstance(block.from_, int) \
                    else [x if j == -1 else cache[j] for j in
                          block.from_]  # calculate concat detect
            x = block(x)
            if i in self.seg_out_idx:  # save driving area segment result
                # m = nn.Sigmoid()
                # out.append(m(x))
                out.append(torch.sigmoid(x))
            if i == self.detector_index:
                # det_out = x
                if self.training:
                    det_out = x
                else:
                    det_out = x[0]  # (torch.cat(z, 1), input_feat) if test
            cache.append(x if block.index in self.save else None)
        return det_out, out[0], out[1]  # det, da, ll
        # (1,na*ny*nx*nl,no=2+2+1+nc=xy+wh+obj_conf+cls_prob), (1,2,h,w) (1,2,h,w)

    def _initialize_biases(self, cf=None):  # initialize biases into Detect(), cf is class frequency
        # https://arxiv.org/abs/1708.02002 section 3.3
        # cf = torch.bincount(torch.tensor(np.concatenate(dataset.labels, 0)[:, 0]).long(), minlength=nc) + 1.
        # m = self.model[-1]  # Detect() module
        m = self.model[self.detector_index]  # Detect() module
        for mi, s in zip(m.m, m.stride):  # from
            b = mi.bias.view(m.na, -1)  # conv.bias(255) to (3,85)
            b[:, 4] += math.log(8 / (640 / s) ** 2)  # obj (8 objects per 640 image)
            b[:, 5:] += math.log(0.6 / (m.nc - 0.99)) if cf is None else torch.log(cf / cf.sum())  # cls
            mi.bias = torch.nn.Parameter(b.view(-1), requires_grad=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--height', type=int, default=640)  # height
    parser.add_argument('--width', type=int, default=640)  # width
    args = parser.parse_args()

    do_simplify = True

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = MCnet(YOLOP)
    checkpoint = torch.load('./weights/End-to-end.pth', map_location=device)
    model.load_state_dict(checkpoint['state_dict'])
    model.eval()

    height = args.height
    width = args.width
    print("Load ./weights/End-to-end.pth done!")
    onnx_path = f'./weights/yolop-{height}-{width}.onnx'
    inputs = torch.randn(1, 3, height, width)

    print(f"Converting to {onnx_path}")
    torch.onnx.export(model, inputs, onnx_path,
                      verbose=False, opset_version=12, input_names=['images'],
                      output_names=['det_out', 'drive_area_seg', 'lane_line_seg'])
    print('convert', onnx_path, 'to onnx finish!!!')
    # Checks
    model_onnx = onnx.load(onnx_path)  # load onnx model
    onnx.checker.check_model(model_onnx)  # check onnx model
    print(onnx.helper.printable_graph(model_onnx.graph))  # print

    if do_simplify:
        print(f'simplifying with onnx-simplifier {onnxsim.__version__}...')
        model_onnx, check = onnxsim.simplify(model_onnx, check_n=3)
        assert check, 'assert check failed'
        onnx.save(model_onnx, onnx_path)

    x = inputs.cpu().numpy()
    try:
        sess = ort.InferenceSession(onnx_path)

        for ii in sess.get_inputs():
            print("Input: ", ii)
        for oo in sess.get_outputs():
            print("Output: ", oo)

        print('read onnx using onnxruntime sucess')
    except Exception as e:
        print('read failed')
        raise e

    """
    PYTHONPATH=. python3 ./export_onnx.py --height 640 --width 640
    PYTHONPATH=. python3 ./export_onnx.py --height 1280 --width 1280
    PYTHONPATH=. python3 ./export_onnx.py --height 320 --width 320
    """

```  

## Python版本onnxruntime测试  

* test_onnx.py  
```python
import os
import cv2
import torch
import argparse
import onnxruntime as ort
import numpy as np
from lib.core.general import non_max_suppression


def resize_unscale(img, new_shape=(640, 640), color=114):
    shape = img.shape[:2]  # current shape [height, width]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    canvas = np.zeros((new_shape[0], new_shape[1], 3))
    canvas.fill(color)
    # Scale ratio (new / old) new_shape(h,w)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])

    # Compute padding
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))  # w,h
    new_unpad_w = new_unpad[0]
    new_unpad_h = new_unpad[1]
    pad_w, pad_h = new_shape[1] - new_unpad_w, new_shape[0] - new_unpad_h  # wh padding

    dw = pad_w // 2  # divide padding into 2 sides
    dh = pad_h // 2

    if shape[::-1] != new_unpad:  # resize
        img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_AREA)

    canvas[dh:dh + new_unpad_h, dw:dw + new_unpad_w, :] = img

    return canvas, r, dw, dh, new_unpad_w, new_unpad_h  # (dw,dh)


def infer_yolop(weight="yolop-640-640.onnx",
                img_path="./inference/images/7dd9ef45-f197db95.jpg"):

    ort.set_default_logger_severity(4)
    onnx_path = f"./weights/{weight}"
    ort_session = ort.InferenceSession(onnx_path)
    print(f"Load {onnx_path} done!")

    outputs_info = ort_session.get_outputs()
    inputs_info = ort_session.get_inputs()

    for ii in inputs_info:
        print("Input: ", ii)
    for oo in outputs_info:
        print("Output: ", oo)

    print("num outputs: ", len(outputs_info))

    save_det_path = f"./pictures/detect_onnx.jpg"
    save_da_path = f"./pictures/da_onnx.jpg"
    save_ll_path = f"./pictures/ll_onnx.jpg"
    save_merge_path = f"./pictures/output_onnx.jpg"

    img_bgr = cv2.imread(img_path)
    height, width, _ = img_bgr.shape

    # convert to RGB
    img_rgb = img_bgr[:, :, ::-1].copy()

    # resize & normalize
    canvas, r, dw, dh, new_unpad_w, new_unpad_h = resize_unscale(img_rgb, (640, 640))

    img = canvas.copy().transpose(2, 0, 1).astype(np.float32)  # (3,640,640) RGB
    img /= 255.0
    img[:, :, 0] -= 0.485
    img[:, :, 1] -= 0.456
    img[:, :, 2] -= 0.406
    img[:, :, 0] /= 0.229
    img[:, :, 1] /= 0.224
    img[:, :, 2] /= 0.225

    img = np.expand_dims(img, 0)  # (1, 3,640,640)

    # inference: (1,n,6) (1,2,640,640) (1,2,640,640)
    det_out, da_seg_out, ll_seg_out = ort_session.run(
        ['det_out', 'drive_area_seg', 'lane_line_seg'],
        input_feed={"images": img}
    )

    det_out = torch.from_numpy(det_out).float()
    boxes = non_max_suppression(det_out)[0]  # [n,6] [x1,y1,x2,y2,conf,cls]
    boxes = boxes.cpu().numpy().astype(np.float32)

    if boxes.shape[0] == 0:
        print("no bounding boxes detected.")
        return

    # scale coords to original size.
    boxes[:, 0] -= dw
    boxes[:, 1] -= dh
    boxes[:, 2] -= dw
    boxes[:, 3] -= dh
    boxes[:, :4] /= r

    print(f"detect {boxes.shape[0]} bounding boxes.")

    img_det = img_rgb[:, :, ::-1].copy()
    for i in range(boxes.shape[0]):
        x1, y1, x2, y2, conf, label = boxes[i]
        x1, y1, x2, y2, label = int(x1), int(y1), int(x2), int(y2), int(label)
        img_det = cv2.rectangle(img_det, (x1, y1), (x2, y2), (0, 255, 0), 2, 2)

    cv2.imwrite(save_det_path, img_det)

    # select da & ll segment area.
    da_seg_out = da_seg_out[:, :, dh:dh + new_unpad_h, dw:dw + new_unpad_w]
    ll_seg_out = ll_seg_out[:, :, dh:dh + new_unpad_h, dw:dw + new_unpad_w]

    da_seg_mask = np.argmax(da_seg_out, axis=1)[0]  # (?,?) (0|1)
    ll_seg_mask = np.argmax(ll_seg_out, axis=1)[0]  # (?,?) (0|1)
    print(da_seg_mask.shape)
    print(ll_seg_mask.shape)

    color_area = np.zeros((new_unpad_h, new_unpad_w, 3), dtype=np.uint8)
    color_area[da_seg_mask == 1] = [0, 255, 0]
    color_area[ll_seg_mask == 1] = [255, 0, 0]
    color_seg = color_area

    # convert to BGR
    color_seg = color_seg[..., ::-1]
    color_mask = np.mean(color_seg, 2)
    img_merge = canvas[dh:dh + new_unpad_h, dw:dw + new_unpad_w, :]
    img_merge = img_merge[:, :, ::-1]

    # merge: resize to original size
    img_merge[color_mask != 0] = \
        img_merge[color_mask != 0] * 0.5 + color_seg[color_mask != 0] * 0.5
    img_merge = img_merge.astype(np.uint8)
    img_merge = cv2.resize(img_merge, (width, height),
                           interpolation=cv2.INTER_LINEAR)
    for i in range(boxes.shape[0]):
        x1, y1, x2, y2, conf, label = boxes[i]
        x1, y1, x2, y2, label = int(x1), int(y1), int(x2), int(y2), int(label)
        img_merge = cv2.rectangle(img_merge, (x1, y1), (x2, y2), (0, 255, 0), 2, 2)

    # da: resize to original size
    da_seg_mask = da_seg_mask * 255
    da_seg_mask = da_seg_mask.astype(np.uint8)
    da_seg_mask = cv2.resize(da_seg_mask, (width, height),
                             interpolation=cv2.INTER_LINEAR)

    # ll: resize to original size
    ll_seg_mask = ll_seg_mask * 255
    ll_seg_mask = ll_seg_mask.astype(np.uint8)
    ll_seg_mask = cv2.resize(ll_seg_mask, (width, height),
                             interpolation=cv2.INTER_LINEAR)

    cv2.imwrite(save_merge_path, img_merge)
    cv2.imwrite(save_da_path, da_seg_mask)
    cv2.imwrite(save_ll_path, ll_seg_mask)

    print("detect done.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--weight', type=str, default="yolop-640-640.onnx")
    parser.add_argument('--img', type=str, default="./inference/images/9aa94005-ff1d4c9a.jpg")
    args = parser.parse_args()

    infer_yolop(weight=args.weight, img_path=args.img)
    """
    PYTHONPATH=. python3 ./test_onnx.py --weight yolop-640-640.onnx --img ./inference/images/9aa94005-ff1d4c9a.jpg
    """

```