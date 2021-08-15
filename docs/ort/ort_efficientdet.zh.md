# 记录EfficientDet工程化

## 项目地址

* [Yet-Another-EfficientDet-Pytorch](https://github.com/zylo117/Yet-Another-EfficientDet-Pytorch)

## 修改ONNX不支持的操作   
在[Yet-Another-EfficientDet-Pytorch](https://github.com/zylo117/Yet-Another-EfficientDet-Pytorch) 中主要有3处onnx不支持的操作。  
* 修改SE操作中的adaptive_avg_pool2d为等价的torch.mean算子  
```python
class MBConvBlock(nn.Module):
    # ... 
    def forward(self, inputs, drop_connect_rate=None):
        # Expansion and Depthwise Convolution
        x = inputs
        if self._block_args.expand_ratio != 1:
            x = self._expand_conv(inputs)
            x = self._bn0(x)
            x = self._swish(x)

        x = self._depthwise_conv(x)
        x = self._bn1(x)
        x = self._swish(x)

        # Squeeze and Excitation
        if self.has_se:
            # x_squeezed = F.adaptive_avg_pool2d(x, 1) onnx不直接支持adaptive_avg_pool2d
            x_squeezed = torch.mean(x, dim=(2, 3), keepdim=True)  # for onnx export
            x_squeezed = self._se_reduce(x_squeezed)
            x_squeezed = self._swish(x_squeezed)
            x_squeezed = self._se_expand(x_squeezed)
            x = torch.sigmoid(x_squeezed) * x

        x = self._project_conv(x)
        x = self._bn2(x)

        # Skip connection and drop connect
        input_filters, output_filters = self._block_args.input_filters, self._block_args.output_filters
        if self.id_skip and self._block_args.stride == 1 and input_filters == output_filters:
            if drop_connect_rate:
                x = drop_connect(x, p=drop_connect_rate, training=self.training)
            x = x + inputs  # skip connection
        return x  
```  
* 将utils_extra.py中的Conv2dStaticSamePadding和MaxPool2dStaticSamePadding修改为静态的。因为事实上根据kernel_size和stride的值，pad对应的4个值的选择是有限的。我们可以用一个全局变量来收集所有的情况，然后再将这些有限的情况全部写出来即可。这样就可以避免根据输入的张量动态地决定pad值、  
  
```python
# Author: Zylo117

import math

import torch
from torch import nn
import torch.nn.functional as F

GLOBAL_PAD_COLLECTION = {}  # 用一个全局变量来收集所有的情况


class Conv2dStaticSamePadding(nn.Module):
    """
    created by Zylo117
    The real keras/tensorflow conv2d with same padding
    """

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, bias=True, groups=1, dilation=1, **kwargs):
        super().__init__()
        if isinstance(stride, int):
            self.stride = tuple([stride] * 2)
        elif len(stride) == 1:
            self.stride = tuple([stride[0]] * 2)

        if kernel_size == 3 and self.stride[0] == 2:
            self.pad = [0, 1, 0, 1]
        elif kernel_size == 3 and self.stride[0] == 1:
            self.pad = [1, 1, 1, 1]
        elif kernel_size == 1 and self.stride[0] == 1:
            self.pad = [0, 0, 0, 0]
        elif kernel_size == 5 and self.stride[0] == 1:
            self.pad = [2, 2, 2, 2]
        elif kernel_size == 5 and self.stride[0] == 2:
            self.pad = [1, 2, 1, 2]
        else:
            raise ValueError(f"unsupport kernel and stride: {kernel_size}, {stride}")

        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride=self.stride,
                              bias=bias, groups=groups)
        self.stride = self.conv.stride
        self.kernel_size = self.conv.kernel_size
        self.dilation = self.conv.dilation

        if isinstance(self.kernel_size, int):
            self.kernel_size = [self.kernel_size] * 2
        elif len(self.kernel_size) == 1:
            self.kernel_size = [self.kernel_size[0]] * 2

        # if isinstance(self.stride, int):
        #     self.stride = [self.stride] * 2
        # elif len(self.stride) == 1:
        #     self.stride = [self.stride[0]] * 2
        #
        # if isinstance(self.kernel_size, int):
        #     self.kernel_size = [self.kernel_size] * 2
        # elif len(self.kernel_size) == 1:
        #     self.kernel_size = [self.kernel_size[0]] * 2

    def forward(self, x):
        # h, w = x.shape[-2:]
        #
        # extra_h = (math.ceil(w / self.stride[1]) - 1) * self.stride[1] - w + self.kernel_size[1]
        # extra_v = (math.ceil(h / self.stride[0]) - 1) * self.stride[0] - h + self.kernel_size[0]

        # # extra_h = torch.scalar_tensor((torch.ceil(w / self.stride[1]) - 1) * self.stride[1]
        # #                               - w + self.kernel_size[1], dtype=torch.long)
        # # extra_v = torch.scalar_tensor((torch.ceil(h / self.stride[0]) - 1) * self.stride[0]
        # #                               - h + self.kernel_size[0], dtype=torch.long)
        #
        # extra_h = (torch.ceil(w / self.stride[1]) - 1) * self.stride[1] - w + self.kernel_size[1]
        # extra_v = (torch.ceil(h / self.stride[0]) - 1) * self.stride[0] - h + self.kernel_size[0]
        print("Conv2dStaticSamePadding Input size: ", x.shape, end=" ")
        print("stride: ", self.stride, "kernel_size: ", self.kernel_size, end=" ")
        # h, w = x.cpu().detach().numpy().shape[-2:]
        #
        # h_step = math.ceil(w / self.stride[1])
        # v_step = math.ceil(h / self.stride[0])
        # h_cover_len = self.stride[1] * (h_step - 1) + 1 + (self.kernel_size[1] - 1)
        # v_cover_len = self.stride[0] * (v_step - 1) + 1 + (self.kernel_size[0] - 1)
        #
        # extra_h = h_cover_len - w
        # extra_v = v_cover_len - h
        # left = extra_h // 2
        # right = extra_h - left
        # top = extra_v // 2
        # bottom = extra_v - top
        #
        # x = F.pad(x, [left, right, top, bottom])

        x = F.pad(x, self.pad)

        print("pads: ", self.pad, end=" ")

        x = self.conv(x)
        print("Conv2dStaticSamePadding Output size: ", x.shape)

        global GLOBAL_PAD_COLLECTION
        GLOBAL_PAD_COLLECTION.update(
            {f"Conv2dStaticSamePadding kernel:{self.kernel_size} strides:{self.stride}": self.pad})

        return x


class MaxPool2dStaticSamePadding(nn.Module):
    """
    created by Zylo117
    The real keras/tensorflow MaxPool2d with same padding
    """

    def __init__(self, *args, **kwargs):
        super().__init__()

        self.pool = nn.MaxPool2d(*args, **kwargs)
        self.stride = self.pool.stride
        self.kernel_size = self.pool.kernel_size

        if isinstance(self.stride, int):
            self.stride = [self.stride] * 2
        elif len(self.stride) == 1:
            self.stride = [self.stride[0]] * 2

        if isinstance(self.kernel_size, int):
            self.kernel_size = [self.kernel_size] * 2
        elif len(self.kernel_size) == 1:
            self.kernel_size = [self.kernel_size[0]] * 2

        if self.kernel_size[0] == 3 and self.stride[0] == 2:
            self.pad = [0, 1, 0, 1]
        else:
            raise ValueError(f"unsupport kernel and stride: {self.kernel_size}, {self.stride}")

    def forward(self, x):
        # h, w = x.shape[-2:]

        # h, w = x.cpu().detach().numpy().shape[-2:]

        print("MaxPool2dStaticSamePadding Input size: ", x.shape, end=" ")
        print("stride: ", self.stride, "kernel_size: ", self.kernel_size, end=" ")
        # h_step = math.ceil(w / self.stride[1])
        # v_step = math.ceil(h / self.stride[0])
        # h_cover_len = self.stride[1] * (h_step - 1) + 1 + (self.kernel_size[1] - 1)
        # v_cover_len = self.stride[0] * (v_step - 1) + 1 + (self.kernel_size[0] - 1)
        #
        # extra_h = h_cover_len - w
        # extra_v = v_cover_len - h

        # extra_h = (math.ceil(w / self.stride[1]) - 1) * self.stride[1] - w + self.kernel_size[1]
        # extra_v = (math.ceil(h / self.stride[0]) - 1) * self.stride[0] - h + self.kernel_size[0]

        # extra_h = torch.scalar_tensor((torch.ceil(w / self.stride[1]) - 1) * self.stride[1]
        #                               - w + self.kernel_size[1], dtype=torch.long)
        # extra_v = torch.scalar_tensor((torch.ceil(h / self.stride[0]) - 1) * self.stride[0]
        #                               - h + self.kernel_size[0], dtype=torch.long)

        # extra_h = (torch.ceil(w / self.stride[1]) - 1) * self.stride[1] - w + self.kernel_size[1]
        # extra_v = (torch.ceil(h / self.stride[0]) - 1) * self.stride[0] - h + self.kernel_size[0]
        #
        # left = extra_h // 2
        # right = extra_h - left
        # top = extra_v // 2
        # bottom = extra_v - top

        # x = F.pad(x, [left, right, top, bottom])
        x = F.pad(x, self.pad)
        print("pads: ", self.pad, end=" ")

        x = self.pool(x)
        print("MaxPool2dStaticSamePadding Output size: ", x.shape)
        GLOBAL_PAD_COLLECTION.update(
            {f"MaxPool2dStaticSamePadding kernel:{self.kernel_size} strides:{self.stride}": self.pad})

        return x

```  
事实上情况是有限的，我们可以在转换成onnx前，将上述两个操作变成静态的pad. 如果不这样做会在onnxruntime调用onnx模型时遇到Conv_35的stride错误问题。 log:  
```shell
{'Conv2dStaticSamePadding kernel:(1, 1) strides:(1, 1)': [0, 0, 0, 0],
 'Conv2dStaticSamePadding kernel:(3, 3) strides:(1, 1)': [1, 1, 1, 1],
 'Conv2dStaticSamePadding kernel:(3, 3) strides:(2, 2)': [0, 1, 0, 1],
 'Conv2dStaticSamePadding kernel:(5, 5) strides:(1, 1)': [2, 2, 2, 2],
 'Conv2dStaticSamePadding kernel:(5, 5) strides:(2, 2)': [1, 2, 1, 2],
 'MaxPool2dStaticSamePadding kernel:[3, 3] strides:[2, 2]': [0, 1, 0, 1]}
Converting to ONNX ...
{'Conv2dStaticSamePadding kernel:(1, 1) strides:(1, 1)': [0, 0, 0, 0],
 'Conv2dStaticSamePadding kernel:(3, 3) strides:(1, 1)': [1, 1, 1, 1],
 'Conv2dStaticSamePadding kernel:(3, 3) strides:(2, 2)': [0, 1, 0, 1],
 'Conv2dStaticSamePadding kernel:(5, 5) strides:(1, 1)': [2, 2, 2, 2],
 'Conv2dStaticSamePadding kernel:(5, 5) strides:(2, 2)': [1, 2, 1, 2],
 'MaxPool2dStaticSamePadding kernel:[3, 3] strides:[2, 2]': [0, 1, 0, 1]}
```    
另外注意要在efficientdet/model.py中修改一下EfficientNet的判断条件：  
```python
class EfficientNet(nn.Module):
  """
  modified by Zylo117
  """

  def __init__(self, compound_coef, load_weights=False):
    super(EfficientNet, self).__init__()
    model = EffNet.from_pretrained(f'efficientnet-b{compound_coef}', load_weights)
    del model._conv_head
    del model._bn1
    del model._avg_pooling
    del model._dropout
    del model._fc
    self.model = model

  def forward(self, x):
    x = self.model._conv_stem(x)
    x = self.model._bn0(x)
    x = self.model._swish(x)
    feature_maps = []

    # TODO: temporarily storing extra tensor last_x and del it later might not be a good idea,
    #  try recording stride changing when creating efficientnet,
    #  and then apply it here.
    last_x = None
    for idx, block in enumerate(self.model._blocks):
      drop_connect_rate = self.model._global_params.drop_connect_rate
      if drop_connect_rate:
        drop_connect_rate *= float(idx) / len(self.model._blocks)
      x = block(x, drop_connect_rate=drop_connect_rate)
      
      # 增加or block._depthwise_conv.stride == (2, 2)
      if block._depthwise_conv.stride == [2, 2] or block._depthwise_conv.stride == (2, 2):
        feature_maps.append(last_x)
      elif idx == len(self.model._blocks) - 1:
        feature_maps.append(x)
      last_x = x
    del last_x
    return feature_maps[1:]
```

* 修改backbone及其他，增加onnx_export参数。同时由于Anchors类中是numpy操作。因此不导出Anchors，只导出regression, classification. 使用c++重新实现等价的anchors生成。
```python

class EfficientDetBackbone(nn.Module):
    def __init__(self, num_classes=80, compound_coef=0, load_weights=False, onnx_export=False, **kwargs):
        super(EfficientDetBackbone, self).__init__()
        self.onnx_export = onnx_export
        self.compound_coef = compound_coef

        self.backbone_compound_coef = [0, 1, 2, 3, 4, 5, 6, 6, 7]
        self.fpn_num_filters = [64, 88, 112, 160, 224, 288, 384, 384, 384]
        self.fpn_cell_repeats = [3, 4, 5, 6, 7, 7, 8, 8, 8]
        self.input_sizes = [512, 640, 768, 896, 1024, 1280, 1280, 1536, 1536]
        self.box_class_repeats = [3, 3, 3, 4, 4, 4, 5, 5, 5]

        self.pyramid_levels = [5, 5, 5, 5, 5, 5, 5, 5, 6]  # d7 & d8
        self.anchor_scale = [4., 4., 4., 4., 4., 4., 4., 5., 4.]  # d7 & d8
        self.aspect_ratios = kwargs.get('ratios', [(1.0, 1.0), (1.4, 0.7), (0.7, 1.4)])
        self.num_scales = len(kwargs.get('scales', [2 ** 0, 2 ** (1.0 / 3.0), 2 ** (2.0 / 3.0)]))
        conv_channel_coef = {
            # the channels of P3/P4/P5.
            0: [40, 112, 320],
            1: [40, 112, 320],
            2: [48, 120, 352],
            3: [48, 136, 384],
            4: [56, 160, 448],
            5: [64, 176, 512],
            6: [72, 200, 576],
            7: [72, 200, 576],
            8: [80, 224, 640],
        }

        num_anchors = len(self.aspect_ratios) * self.num_scales

        self.bifpn = nn.Sequential(
            *[BiFPN(self.fpn_num_filters[self.compound_coef],
                    conv_channel_coef[compound_coef],
                    True if _ == 0 else False,
                    attention=True if compound_coef < 6 else False,
                    use_p8=compound_coef > 7,
                    onnx_export=onnx_export)
              for _ in range(self.fpn_cell_repeats[compound_coef])])

        self.num_classes = num_classes
        self.regressor = Regressor(in_channels=self.fpn_num_filters[self.compound_coef],
                                   num_anchors=num_anchors,
                                   num_layers=self.box_class_repeats[self.compound_coef],
                                   pyramid_levels=self.pyramid_levels[self.compound_coef],
                                   onnx_export=onnx_export)
        self.classifier = Classifier(in_channels=self.fpn_num_filters[self.compound_coef],
                                     num_anchors=num_anchors,
                                     num_classes=num_classes,
                                     num_layers=self.box_class_repeats[self.compound_coef],
                                     pyramid_levels=self.pyramid_levels[self.compound_coef],
                                     onnx_export=onnx_export)

        self.anchors = Anchors(anchor_scale=self.anchor_scale[compound_coef],
                               pyramid_levels=(torch.arange(self.pyramid_levels[self.compound_coef]) + 3).tolist(),
                               **kwargs)

        self.backbone_net = EfficientNet(self.backbone_compound_coef[compound_coef], load_weights)

    def freeze_bn(self):
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()

    def forward(self, inputs):
        # max_size = inputs.shape[-1]

        _, p3, p4, p5 = self.backbone_net(inputs)

        features = (p3, p4, p5)
        features = self.bifpn(features)

        regression = self.regressor(features)  # (batch,num_boxes,4) (dy, dx, dh, dw)]
        classification = self.classifier(features)  # (batch,num_boxes,num_classes) 每个类都是0-1单独分类

        # export regression, classification for onnx.
        if self.onnx_export:
            return regression, classification

        anchors = self.anchors(inputs, inputs.dtype)  # (num_boxes,4) (y1, x1, y2, x2)]
        return features, regression, classification, anchors

    def init_backbone(self, path):
        state_dict = torch.load(path)
        try:
            ret = self.load_state_dict(state_dict, strict=False)
            print(ret)
        except RuntimeError as e:
            print('Ignoring ' + str(e) + '"')

```  
关于Anchors的C++等价实现。以下首先是对Python版本的理解和注释。  
```python
class Anchors(nn.Module):
    """
    adapted and modified from https://github.com/google/automl/blob/master/efficientdet/anchors.py by Zylo117
    """

    def __init__(self, anchor_scale=4., pyramid_levels=None, **kwargs):
        super().__init__()
        self.anchor_scale = anchor_scale

        if pyramid_levels is None:
            self.pyramid_levels = [3, 4, 5, 6, 7]
        else:
            self.pyramid_levels = pyramid_levels

        self.strides = kwargs.get('strides', [2 ** x for x in self.pyramid_levels])
        self.scales = np.array(kwargs.get('scales', [2 ** 0, 2 ** (1.0 / 3.0), 2 ** (2.0 / 3.0)]))
        self.ratios = kwargs.get('ratios', [(1.0, 1.0), (1.4, 0.7), (0.7, 1.4)])

        self.last_anchors = {}
        self.last_shape = None

    def forward(self, image, dtype=torch.float32):
        """Generates multiscale anchor boxes.

        Args:
          image_size: integer number of input image size. The input image has the
            same dimension for width and height. The image_size should be divided by
            the largest feature stride 2^max_level.
          anchor_scale: float number representing the scale of size of the base
            anchor to the feature stride 2^level.
          anchor_configs: a dictionary with keys as the levels of anchors and
            values as a list of anchor configuration.

        Returns:
          anchor_boxes: a numpy array with shape [N, 4], which stacks anchors on all
            feature levels.
        Raises:
          ValueError: input size must be the multiple of largest feature stride.
        """
        image_shape = image.shape[2:]

        if image_shape == self.last_shape and image.device in self.last_anchors:
            return self.last_anchors[image.device]

        if self.last_shape is None or self.last_shape != image_shape:
            self.last_shape = image_shape

        if dtype == torch.float16:
            dtype = np.float16
        else:
            dtype = np.float32

        boxes_all = []
        # strides=[8,16,32,64,128] | [8,16,32,64,128,256]
        # scales=[2 ** 0, 2 ** (1.0 / 3.0), 2 ** (2.0 / 3.0)])
        # ratios=[(1.0, 1.0), (1.4, 0.7), (0.7, 1.4)])
        # 即对于每一个步长下的特征图，每个锚点的anchor有3个不同的尺度scale，
        # 每个尺度又有3个不同的宽高比率ratio; 即每个锚点对应3x3=9个anchors
        # anchor_scale是一个常数 用于控制基准的宽高
        for stride in self.strides:
            boxes_level = []
            for scale, ratio in itertools.product(self.scales, self.ratios):
                """
                scale=2 ** 0, ratio=(1.0,1.0)
                scale=2 ** 0, ratio=(1.4,0.7)
                scale=2 ** 0, ratio=(0.7,1.4)
                scale=2 ** (1.0 / 3.0), ratio=(1.0,1.0)
                scale=2 ** (1.0 / 3.0), ratio=(1.4,0.7)
                scale=2 ** (1.0 / 3.0), ratio=(0.7,1.4)
                scale=2 ** (2.0 / 3.0), ratio=(1.0,1.0)
                scale=2 ** (2.0 / 3.0), ratio=(1.4,0.7)
                scale=2 ** (2.0 / 3.0), ratio=(0.7,1.4)
                """
                if image_shape[1] % stride != 0:
                    raise ValueError('input size must be divided by the stride.')
                base_anchor_size = self.anchor_scale * stride * scale
                # 特征图的点对应的anchor的大小的一半 对应到输入图尺寸 宽高有各自的比率ratio
                anchor_size_x_2 = base_anchor_size * ratio[0] / 2.0
                anchor_size_y_2 = base_anchor_size * ratio[1] / 2.0

                # 这段逻辑ONNX不一定能转换 需要修改成torch运算 # 步长为stride
                """
                >>> np.arange(8/2, 512, 8)
                array([  4.,  12.,  20.,  28.,  36.,  44.,  52.,  60.,  68.,  76.,  84.,
                        92., 100., 108., 116., 124., 132., 140., 148., 156., 164., 172.,
                       180., 188., 196., 204., 212., 220., 228., 236., 244., 252., 260.,
                       268., 276., 284., 292., 300., 308., 316., 324., 332., 340., 348.,
                       356., 364., 372., 380., 388., 396., 404., 412., 420., 428., 436.,
                       444., 452., 460., 468., 476., 484., 492., 500., 508.])
                """
                x = np.arange(stride / 2, image_shape[1], stride)
                y = np.arange(stride / 2, image_shape[0], stride)  # 特征图的点在输入图上对应的位置 锚点中心
                xv, yv = np.meshgrid(x, y)
                xv = xv.reshape(-1)
                yv = yv.reshape(-1)

                # y1,x1,y2,x2  y1=cy-ch/2 x1=cx-cw/2 y2=cy+ch/2 x1=cx+cw/2
                boxes = np.vstack((yv - anchor_size_y_2, xv - anchor_size_x_2,
                                   yv + anchor_size_y_2, xv + anchor_size_x_2))  # (4,N)
                boxes = np.swapaxes(boxes, 0, 1)  # (N,4)
                boxes_level.append(np.expand_dims(boxes, axis=1))  # append (N,1,4) expand_dims是否多余？
            # concat anchors on the same level to the reshape NxAx4
            # 因为此时步长是固定的所以boxes的个数是相同的 N=N0=N1=...  
            # 在一个stride下 同一个锚点的9个anchor安顺序叠在一起
            boxes_level = np.concatenate(boxes_level, axis=1)  # [N,A=9,4] [(N,1,4),(N,1,4),...]
            boxes_all.append(boxes_level.reshape([-1, 4]))  # [NX9,4]

        anchor_boxes = np.vstack(boxes_all)  # 每个步长一个boxes_all [5XNX9,4]

        anchor_boxes = torch.from_numpy(anchor_boxes.astype(dtype)).to(image.device)  # Constant for specific
        anchor_boxes = anchor_boxes.unsqueeze(0)  # [1,5XNX9,4]

        # save it for later use to reduce overhead
        self.last_anchors[image.device] = anchor_boxes
        return anchor_boxes

```  
而C++的等价实现为:   
```c++  

// ref: https://github.com/zylo117/Yet-Another-EfficientDet-Pytorch/blob/master/efficientdet/utils.py
void EfficientDet::generate_anchors(const float target_height, const float target_width)
{
  if (!anchors_buffer.empty()) return;

  // generate once.
  for (const auto &stride: strides)
  {
    // create grid with a specific stride. Under a specific stride,
    // 9 Anchors of the same anchor point are stacked together in order
    for (float yv = stride / 2.0f; yv < target_height; yv += stride)
    {
      for (float xv = stride / 2.0f; xv < target_width; xv += stride)
      {
        for (const auto &scale: scales)
        {
          for (const auto &ratio: ratios)
          {
            float base_anchor_size = anchor_scale * stride * scale;
            // aw/2 and ah/2, according to input size with different ratio.
            float anchor_size_x_2 = base_anchor_size * ratio[0] / 2.0f;
            float anchor_size_y_2 = base_anchor_size * ratio[1] / 2.0f;

            float y1 = yv - anchor_size_y_2; // cy - ah/2
            float x1 = xv - anchor_size_x_2; // cx - aw/2
            float y2 = yv + anchor_size_y_2; // cy + ah/2
            float x2 = xv + anchor_size_x_2; // cx + aw/2

            anchors_buffer.push_back((EfficientDetAnchor) {y1, x1, y2, x2});
          } // end ratios 3
        } // end scale 3
      }
    } // end grid
  } // end strides
}

```  

## 转换为ONNX文件  
```python
import torch
import onnx
import onnxruntime as ort
import yaml
from torch import nn
from backbone import EfficientDetBackbone
import numpy as np
from efficientnet.utils_extra import GLOBAL_PAD_COLLECTION
from pprint import pprint

obj_list = ['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light',
            'fire hydrant', '', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep',
            'cow', 'elephant', 'bear', 'zebra', 'giraffe', '', 'backpack', 'umbrella', '', '', 'handbag', 'tie',
            'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove',
            'skateboard', 'surfboard', 'tennis racket', 'bottle', '', 'wine glass', 'cup', 'fork', 'knife', 'spoon',
            'bowl', 'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut',
            'cake', 'chair', 'couch', 'potted plant', 'bed', '', 'dining table', '', '', 'toilet', '', 'tv',
            'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink',
            'refrigerator', '', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier',
            'toothbrush']


class Params:
    def __init__(self, project_file):
        self.params = yaml.safe_load(open(project_file).read())

    def __getattr__(self, item):
        return self.params.get(item, None)


def convert_to_onnx(pretrained_path='./weights/efficientdet-d0.pth',
                    onnx_path='./weights/efficientdet-d0.onnx',
                    compound_coef=0, sim_only=False):

    if not sim_only:
        params = Params(f'projects/coco.yml')
        obj_list = params.obj_list
        # replace this part with your project's anchor config
        # anchor_ratios = [(1.0, 1.0), (1.4, 0.7), (0.7, 1.4)]
        # anchor_scales = [2 ** 0, 2 ** (1.0 / 3.0), 2 ** (2.0 / 3.0)]

        anchor_ratios = eval(params.anchors_ratios)
        anchor_scales = eval(params.anchors_scales)
        input_sizes = [512, 640, 768, 896, 1024, 1280, 1280, 1536, 1536]
        input_size = input_sizes[compound_coef]
        num_classes = len(obj_list)

        print("num_classes: ", num_classes)
        print("anchor_ratios: ", anchor_ratios)
        print("anchor_scales: ", anchor_scales)

        device = torch.device('cpu')

        model = EfficientDetBackbone(num_classes=num_classes,
                                     compound_coef=compound_coef,
                                     onnx_export=True,
                                     load_weights=False,
                                     ratios=anchor_ratios,
                                     scales=anchor_scales).to(device)

        model.backbone_net.model.set_swish(memory_efficient=False)

        model.load_state_dict(torch.load(pretrained_path, map_location=device))

        model.requires_grad_(False)
        model.eval()

        # print(model)

        print(f"Load {pretrained_path} done ! Device: {device}. ")

        dummy_input = torch.randn((1, 3, input_size, input_size), dtype=torch.float32).to(device)
        # opset_version can be changed to 10 or other number, based on your need

        y = model(dummy_input)

        print(y[0].shape)
        print(y[1].shape)
        # print(y[2].shape)

        pprint(GLOBAL_PAD_COLLECTION)

        print("Converting to ONNX ...")
        pprint(GLOBAL_PAD_COLLECTION)

        torch.onnx.export(model,
                          dummy_input,
                          onnx_path,
                          verbose=True,
                          input_names=['data'],
                          output_names=["regression", "classification"],
                          opset_version=11)

        # Checks
        onnx_model = onnx.load(onnx_path)  # load onnx model
        onnx.checker.check_model(onnx_model)  # check onnx model
        print(onnx.helper.printable_graph(onnx_model.graph))  # print a human readable model

        print("Check done!")

        do_simplify = False
        if do_simplify:
            from onnxsim import simplify

            onnx_model, check = simplify(onnx_model, check_n=1)
            assert check, 'assert simplify check failed'
            onnx.save(onnx_model, onnx_path)
            onnx_model = onnx.load(onnx_path)  # load onnx model
            onnx.checker.check_model(onnx_model)  # check onnx model
            print(onnx.helper.printable_graph(onnx_model.graph))  # print a human readable model

            print("Check Sim done!")

        session = ort.InferenceSession(onnx_path)

        for ii in session.get_inputs():
            print("input: ", ii)

        for oo in session.get_outputs():
            print("output: ", oo)

        print('ONNX export success, saved as %s' % onnx_path)

    else:
        # Checks
        onnx_model = onnx.load(onnx_path)  # load onnx model
        onnx.checker.check_model(onnx_model)  # check onnx model
        print(onnx.helper.printable_graph(onnx_model.graph))  # print a human readable model

        # from onnx import optimizer
        #
        # op = ["eliminate_unused_initializer"]
        # onnx_model = optimizer.optimize(onnx_model, None)
        # onnx.save(onnx_model, "efficientdet-d0-opt.onnx")

        print("Check done!")

        do_simplify = True
        if do_simplify:
            from onnxsim import simplify

            onnx_model, check = simplify(onnx_model, check_n=1)
            assert check, 'assert simplify check failed'
            onnx.save(onnx_model, onnx_path)
            onnx_model = onnx.load(onnx_path)  # load onnx model
            onnx.checker.check_model(onnx_model)  # check onnx model
            print(onnx.helper.printable_graph(onnx_model.graph))  # print a human readable model

            print("Check Sim done!")

        session = ort.InferenceSession(onnx_path)

        for ii in session.get_inputs():
            print("input: ", ii)

        for oo in session.get_outputs():
            print("output: ", oo)

        print('ONNX export success, saved as %s' % onnx_path)


if __name__ == "__main__":
    convert_to_onnx(pretrained_path='./weights/efficientdet-d0.pth',
                    onnx_path='./weights/efficientdet-d0.onnx',
                    compound_coef=0)
    convert_to_onnx(pretrained_path='./weights/efficientdet-d1.pth',
                    onnx_path='./weights/efficientdet-d1.onnx',
                    compound_coef=1)
    convert_to_onnx(pretrained_path='./weights/efficientdet-d2.pth',
                    onnx_path='./weights/efficientdet-d2.onnx',
                    compound_coef=2)
    convert_to_onnx(pretrained_path='./weights/efficientdet-d3.pth',
                    onnx_path='./weights/efficientdet-d3.onnx',
                    compound_coef=3)
    convert_to_onnx(pretrained_path='./weights/efficientdet-d4.pth',
                    onnx_path='./weights/efficientdet-d4.onnx',
                    compound_coef=4)
    convert_to_onnx(pretrained_path='./weights/efficientdet-d5.pth',
                    onnx_path='./weights/efficientdet-d5.onnx',
                    compound_coef=5)
    convert_to_onnx(pretrained_path='./weights/efficientdet-d6.pth',
                    onnx_path='./weights/efficientdet-d6.onnx',
                    compound_coef=6)
    convert_to_onnx(pretrained_path='./weights/efficientdet-d7.pth',
                    onnx_path='./weights/efficientdet-d7.onnx',
                    compound_coef=7)
    convert_to_onnx(pretrained_path='./weights/efficientdet-d8.pth',
                    onnx_path='./weights/efficientdet-d8.onnx',
                    compound_coef=8)
    """
    PYTHONPATH=. python3 ./convert_to_onnx.py
    """

```  

## 测试ONNX模型结果  
```python
import time
import torch
import onnx
import onnxruntime as ort
import cv2
import numpy as np
import yaml
import os

from efficientdet.utils import BBoxTransform, ClipBoxes
from utils.utils import preprocess, postprocess, \
    STANDARD_COLORS, standard_to_bgr, get_index_label, \
    plot_one_box, aspectaware_resize_padding

from efficientdet.utils import AnchorsV2
from typing import Union

threshold = 0.2
iou_threshold = 0.2


class Params:
    def __init__(self, project_file):
        self.params = yaml.safe_load(open(project_file).read())

    def __getattr__(self, item):
        return self.params.get(item, None)


def display(preds, imgs, obj_list, compound_coef=0, imshow=False, imwrite=False):
    color_list = standard_to_bgr(STANDARD_COLORS)

    for i in range(len(imgs)):
        if len(preds[i]['rois']) == 0:
            continue

        imgs[i] = imgs[i].copy()

        for j in range(len(preds[i]['rois'])):
            (x1, y1, x2, y2) = preds[i]['rois'][j].astype(np.int)
            obj = obj_list[preds[i]['class_ids'][j]]
            score = float(preds[i]['scores'][j])

            plot_one_box(imgs[i], [x1, y1, x2, y2], label=obj, score=score,
                         color=color_list[get_index_label(obj, obj_list)])
        if imshow:
            cv2.imshow('img', imgs[i])
            cv2.waitKey(0)

        if imwrite:
            os.makedirs('test/', exist_ok=True)
            cv2.imwrite(f'test/img_onnx_{compound_coef}.jpg', imgs[i])
            print(f"Saved img_onnx_{compound_coef}.jpg")


def invert_affine(old_h, old_w, new_h, new_w, preds):
    for i in range(len(preds)):
        if len(preds[i]['rois']) == 0:
            continue
        else:
            # x1,y1.x2.y2
            preds[i]['rois'][:, [0, 2]] = preds[i]['rois'][:, [0, 2]] / (new_w / old_w)
            preds[i]['rois'][:, [1, 3]] = preds[i]['rois'][:, [1, 3]] / (new_h / old_h)
    return preds


def infer_onnx(compound_coef):
    onnx_path = f"./weights/efficientdet-d{compound_coef}.onnx"

    params = Params(f'projects/coco.yml')
    obj_list = params.obj_list
    # replace this part with your project's anchor config
    # anchor_ratios = [(1.0, 1.0), (1.4, 0.7), (0.7, 1.4)]
    # anchor_scales = [2 ** 0, 2 ** (1.0 / 3.0), 2 ** (2.0 / 3.0)]

    anchor_ratios = eval(params.anchors_ratios)
    anchor_scales = eval(params.anchors_scales)
    input_sizes = [512, 640, 768, 896, 1024, 1280, 1280, 1536, 1536]
    pyramid_levels = [5, 5, 5, 5, 5, 5, 5, 5, 6]
    anchor_scale = [4., 4., 4., 4., 4., 4., 4., 5., 4.]
    input_size = input_sizes[compound_coef]
    num_classes = len(obj_list)

    session = ort.InferenceSession(onnx_path)

    print(f"Loaded {onnx_path} done.")

    anchors_func = AnchorsV2(input_size=input_size,
                             anchor_scale=anchor_scale[compound_coef],
                             pyramid_levels=(torch.arange(pyramid_levels[compound_coef]) + 3).tolist())

    regressBoxes = BBoxTransform()
    clipBoxes = ClipBoxes()

    anchors = anchors_func()
    print("anchors.shape: ", anchors.shape)
    print("anchors.data[:100, :]")
    print(anchors.cpu().numpy()[0][:100, :])

    for ii in session.get_inputs():
        print("input: ", ii)

    for oo in session.get_outputs():
        print("output: ", oo)

    img_path = "./test/img.png"
    img = cv2.imread(img_path)
    old_h, old_w, _ = img.shape

    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # canvas, new_w, new_h, old_w, old_h, padding_w, padding_h = \
    #     aspectaware_resize_padding(image=img_rgb, width=input_size, height=input_size)

    img_rgb = cv2.resize(img_rgb, (input_size, input_size))
    img_rgb = img_rgb.astype(np.float32) / 255.0
    img_rgb[:, :, 0] -= 0.406
    img_rgb[:, :, 1] -= 0.456
    img_rgb[:, :, 2] -= 0.485
    img_rgb[:, :, 0] /= 0.225
    img_rgb[:, :, 1] /= 0.224
    img_rgb[:, :, 2] /= 0.229
    img_rgb = np.transpose(img_rgb, (2, 0, 1))
    x = np.expand_dims(img_rgb, axis=0)  # (1,3,512,512)

    regression, classification = session.run(["regression", "classification"],
                                             input_feed={"data": x})

    print("regression[0][:100, :]")
    print(regression[0][:100, :])

    regression = torch.from_numpy(regression)
    classification = torch.from_numpy(classification)
    print("regression: ", regression.shape)
    print("classification: ", classification.shape)

    out = postprocess(x,
                      anchors, regression, classification,
                      regressBoxes, clipBoxes,
                      threshold, iou_threshold)  # rois xmin, ymin, xmax, ymax
    out = invert_affine(old_h=old_h, old_w=old_w, new_h=input_size, new_w=input_size, preds=out)

    display(preds=out, imgs=[img], obj_list=obj_list, compound_coef=compound_coef, imwrite=True)

    print(f"Infer {onnx_path} done !")


if __name__ == "__main__":
    np.set_printoptions(suppress=True)
    infer_onnx(compound_coef=0)
    infer_onnx(compound_coef=1)
    infer_onnx(compound_coef=2)
    infer_onnx(compound_coef=3)
    infer_onnx(compound_coef=4)
    infer_onnx(compound_coef=5)
    infer_onnx(compound_coef=6)
    infer_onnx(compound_coef=7)
    infer_onnx(compound_coef=8)

    """
    PYTHONPATH=. python3 ./inference.py
    """
```  
## C++推理完整工程   
[efficientdet.cpp](https://github.com/DefTruth/lite.ai/blob/main/ort/cv/efficientdet.cpp)