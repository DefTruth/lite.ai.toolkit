# 记录TinyYoloV4工程化  

## 项目地址 

* [yolov4-tiny-pytorch](https://github.com/bubbliiiing/yolov4-tiny-pytorch)

## 修改ONNX不支持的操作  
```python

class DecodeBox(nn.Module):
    def __init__(self, anchors, num_classes, img_size):
        super(DecodeBox, self).__init__()
        self.anchors = anchors
        self.num_anchors = len(anchors)

        self.anchors = torch.from_numpy(self.anchors).type(torch.FloatTensor)
        print(self.anchors.dtype)
        self.num_classes = num_classes
        self.bbox_attrs = 5 + num_classes
        self.img_size = img_size

    def forward(self, input):
        # -----------------------------------------------#
        #   输入的input一共有两个，他们的shape分别是
        #   batch_size, 255, 13, 13
        #   batch_size, 255, 26, 26
        # -----------------------------------------------#
        batch_size = input.size(0)
        input_height = input.size(2)  # 特征图的高
        input_width = input.size(3)  # 特征图的宽

        # -----------------------------------------------#
        #   输入为416x416时
        #   stride_h = stride_w = 32、16
        # -----------------------------------------------#
        stride_h = torch.scalar_tensor(self.img_size[1] / input_height).type(torch.FloatTensor)
        stride_w = torch.scalar_tensor(self.img_size[0] / input_width).type(torch.FloatTensor)
        # -------------------------------------------------#
        #   此时获得的scaled_anchors大小是相对于特征层的
        # -------------------------------------------------#
        # scaled_anchors = [(anchor_width / stride_w, anchor_height / stride_h)
        #                   for anchor_width, anchor_height in
        #                   self.anchors]

        # scaled_anchors = self.anchors

        aw = self.anchors[:, 0:1] / stride_w
        ah = self.anchors[:, 1:2] / stride_h

        scaled_anchors = torch.cat((aw, ah), -1)

        # -----------------------------------------------#
        #   输入的input一共有两个，他们的shape分别是
        #   batch_size, 3, 13, 13, 85
        #   batch_size, 3, 26, 26, 85
        # -----------------------------------------------#

        # 网格上每个点有3个anchor 每个anchor预测 中心偏移量和宽高调整参数
        prediction = input.view(batch_size, self.num_anchors,
                                self.bbox_attrs, input_height,
                                input_width)
        prediction = prediction.permute(0, 1, 3, 4, 2).contiguous()  # [b, 3, 13, 13, 85]
        # prediction = prediction.(0, 1, 3, 4, 2)

        # # 先验框的中心位置的调整参数
        # x = torch.sigmoid(prediction[..., 0])  # (1,3,13,13)
        # y = torch.sigmoid(prediction[..., 1])  # (1,3,13,13)
        # # 先验框的宽高调整参数
        # w = prediction[..., 2]
        # h = prediction[..., 3]
        # # 获得置信度，是否有物体
        # conf = torch.sigmoid(prediction[..., 4])
        # # 种类置信度
        # pred_cls = torch.sigmoid(prediction[..., 5:])

        x = torch.sigmoid(prediction[..., 0:1])  # (1,3,13,13,1)
        y = torch.sigmoid(prediction[..., 1:2])  # (1,3,13,13,1)
        # 先验框的宽高调整参数
        w = prediction[..., 2:3]
        h = prediction[..., 3:4]
        # 获得置信度，是否有物体
        conf = torch.sigmoid(prediction[..., 4:5])
        # 种类置信度
        pred_cls = torch.sigmoid(prediction[..., 5:])

        # LongTensor = torch.LongTensor

        # ----------------------------------------------------------#
        #   生成网格，先验框中心，网格左上角 
        #   batch_size,3,13,13
        # ----------------------------------------------------------#

        # (0.,...,12.) | (0.,...,25.) onnx不支持linspace算子 修改成torch.arange
        # grid_x = torch.linspace(0, input_width - 1, input_width).repeat(input_height, 1).repeat(
        #     batch_size * self.num_anchors, 1, 1).view(x.shape)  #
        # grid_y = torch.linspace(0, input_height - 1, input_height).repeat(input_width, 1).t().repeat(
        #     batch_size * self.num_anchors, 1, 1).view(y.shape)
        grid_x = torch.arange(0, input_width, 1).repeat(input_height, 1).repeat(
            batch_size * self.num_anchors, 1, 1).view(x.shape).type(torch.FloatTensor)  #
        grid_y = torch.arange(0, input_height, 1).repeat(input_width, 1).t().repeat(
            batch_size * self.num_anchors, 1, 1).view(y.shape).type(torch.FloatTensor)
        # grid_y, grid_x = torch.meshgrid([torch.arange(input_height), torch.arange(input_width)])
        # print(torch.linspace(0, input_width - 1, input_width))
        # print(torch.arange(0, input_width, 1))
        # ----------------------------------------------------------#
        #   按照网格格式生成先验框的宽高
        #   batch_size,3,13,13
        # ----------------------------------------------------------#
        # anchor_w = FloatTensor(scaled_anchors).index_select(1, LongTensor([0]))
        # anchor_h = FloatTensor(scaled_anchors).index_select(1, LongTensor([1]))

        anchor_w = scaled_anchors[:, 0:1]
        anchor_h = scaled_anchors[:, 1:2]
        # anchor_h = FloatTensor(scaled_anchors).index_select(1, LongTensor([1]))
        anchor_w = anchor_w.repeat(batch_size, 1).repeat(1, 1, input_height * input_width).view(w.shape)
        anchor_h = anchor_h.repeat(batch_size, 1).repeat(1, 1, input_height * input_width).view(h.shape)

        # ----------------------------------------------------------#
        #   利用预测结果对先验框进行调整
        #   首先调整先验框的中心，从先验框中心向右下角偏移
        #   再调整先验框的宽高。
        # ----------------------------------------------------------#
        cx = (x + grid_x) * stride_w  # 将原有的pytorch直接内存操作修改成常规操作 ONNX虽然转换时没问题 但运行时效果无法对齐 ONNX似乎不支持这种直接的内存操作
        cy = (y + grid_y) * stride_h
        cw = (torch.exp(w) * anchor_w) * stride_w
        ch = (torch.exp(h) * anchor_h) * stride_h
        pred_boxes = torch.cat((cx, cy, cw, ch), -1)

        output = torch.cat((pred_boxes.view(batch_size, -1, 4),
                            conf.view(batch_size, -1, 1),
                            pred_cls.view(batch_size, -1, self.num_classes)), 2)

        return output  # [b, n, 85]
```  

## 将DecodeBox模块一起导出  

补充一下进展，已经成功转换至onnx啦~ 把DecodeBox中一些onnx不支持的算子替换掉，以及将直接内存操作换成常规操作，可以转成onnx，结果也对齐啦~ 看起来效果不错。
* 增加一个类YoloBodyV2，连带DecodeBox一起导出成onnx  

```python  
class YoloBodyV2(nn.Module):
    def __init__(self, anchors, num_anchors, img_size, num_classes, phi=0):
        from utils.utils import DecodeBox
        # num_anchor=3 num_classes=80|20
        super(YoloBodyV2, self).__init__()
        self.anchors = anchors  # (6,2)
        self.anchors_mask = [[3, 4, 5], [1, 2, 3]]
        self.img_size = img_size
        self.anchors0 = self.anchors[self.anchors_mask[0]]
        self.anchors1 = self.anchors[self.anchors_mask[1]]
        if phi >= 4:
            raise AssertionError("Phi must be less than or equal to 3 (0, 1, 2, 3).")

        self.yolo_body = YoloBody(num_anchors, num_classes, phi)
        self.yolo_decode0 = DecodeBox(self.anchors0, num_classes, self.img_size)  # 修改后的DecodeBox
        self.yolo_decode1 = DecodeBox(self.anchors1, num_classes, self.img_size)

    def forward(self, x):
        out0, out1 = self.yolo_body(x)

        pred0 = self.yolo_decode0(out0)  # (b,3,85,13,13)
        pred1 = self.yolo_decode1(out1) # (b,3,85,26,26)
        pred = torch.cat([pred0, pred1], dim=1)  # (b,num_anchors=n+m,4)
        return pred
```  

* 导出成onnx ，并测试效果  

```python  
import os
import cv2
import onnx
import torch
import numpy as np
import onnxruntime as ort
from onnxsim import simplify
from nets.yolo4_tiny import YoloBodyV2
from utils.utils import non_max_suppression


def _get_class(classes_path):
    classes_path = os.path.expanduser(classes_path)
    with open(classes_path) as f:
        class_names = f.readlines()
    class_names = [c.strip() for c in class_names]
    return class_names  # [80,] for coco [20,] for voc


def _get_anchors(anchors_path):
    anchors_path = os.path.expanduser(anchors_path)
    with open(anchors_path) as f:
        anchors = f.readline()
    anchors = [float(x) for x in anchors.split(',')]
    return np.reshape(np.array(anchors).reshape([-1, 3, 2]), [-1, 2])  # (6,2)


def convert_to_onnx(model_path="./model_data/yolov4_tiny_weights_coco.pth",
                    anchors_path="./model_data/yolo_anchors.txt",
                    classes_path="model_data/coco_classes.txt",
                    onnx_path="./model_data/yolov4_tiny_weights_coco.onnx",
                    phi=0,
                    model_image_size=(416, 416, 3),
                    confidence=0.5,
                    iou=0.3):
    class_names = _get_class(classes_path=classes_path)
    anchors = _get_anchors(anchors_path=anchors_path)

    HEIGHT = model_image_size[0]  # 模型输入
    WIDTH = model_image_size[1]

    num_anchors = 3
    num_classes = len(class_names)  # 80 | 20

    net = YoloBodyV2(anchors=anchors,
                     num_anchors=num_anchors,
                     num_classes=num_classes,
                     img_size=(WIDTH, HEIGHT),  # w,h
                     phi=phi).eval()
    print('Loading weights into state dict...')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    state_dict = torch.load(model_path, map_location=device)
    net.yolo_body.load_state_dict(state_dict)
    net = net.eval()
    print('Finished!')

    test_path = "./img/street.jpg"
    img = cv2.imread(test_path)
    img_copy = img.copy()

    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    image_shape = img_rgb.shape[0:2]

    img_rgb = cv2.resize(img_rgb, (WIDTH, HEIGHT))
    img_rgb = img_rgb.astype(np.float32) / 255.0
    img_rgb = np.transpose(img_rgb, (2, 0, 1))
    x = np.expand_dims(img_rgb, axis=0)
    x_tensor = torch.from_numpy(x)

    output_ = net(x_tensor)

    print("output_.shape: ", output_.shape)

    batch_detections = non_max_suppression(output_, len(class_names),
                                           conf_thres=confidence,
                                           nms_thres=iou)
    batch_detections = batch_detections[0].detach().cpu().numpy()

    # ---------------------------------------------------------#
    #   对预测框进行得分筛选
    # ---------------------------------------------------------#
    top_index = batch_detections[:, 4] * batch_detections[:, 5] > confidence
    top_conf = batch_detections[top_index, 4] * batch_detections[top_index, 5]
    top_label = np.array(batch_detections[top_index, -1], np.int32)
    top_bboxes = np.array(batch_detections[top_index, :4])
    top_xmin, top_ymin, top_xmax, top_ymax = np.expand_dims(top_bboxes[:, 0], -1), np.expand_dims(
        top_bboxes[:, 1], -1), np.expand_dims(top_bboxes[:, 2], -1), np.expand_dims(top_bboxes[:, 3], -1)

    top_xmin = top_xmin / model_image_size[1] * image_shape[1]  # width
    top_ymin = top_ymin / model_image_size[0] * image_shape[0]
    top_xmax = top_xmax / model_image_size[1] * image_shape[1]
    top_ymax = top_ymax / model_image_size[0] * image_shape[0]
    boxes = np.concatenate([top_ymin, top_xmin, top_ymax, top_xmax], axis=-1)
    print(boxes)

    for i, c in enumerate(top_label):
        predicted_class = class_names[c]
        score = top_conf[i]

        top, left, bottom, right = boxes[i]
        top = top - 5
        left = left - 5
        bottom = bottom + 5
        right = right + 5

        top = max(0, np.floor(top + 0.5).astype('int32'))
        left = max(0, np.floor(left + 0.5).astype('int32'))
        bottom = min(np.shape(img)[0], np.floor(bottom + 0.5).astype('int32'))
        right = min(np.shape(img)[1], np.floor(right + 0.5).astype('int32'))

        # 画框框
        label = '{} {:.2f}'.format(predicted_class, score)

        cv2.putText(img, label, (left, top), 1, 1, (0, 255, 0), 1)
        cv2.rectangle(img, (left, top), (right, bottom), (0, 0, 255), 1)

    cv2.imwrite("./img/street_test_v2.jpg", img)
    print("Test pth done!")

    # convert
    print("Converting to onnx ...")
    torch.onnx.export(net,
                      x_tensor,
                      onnx_path,
                      input_names=["input"],
                      output_names=["output"],
                      opset_version=11,
                      export_params=True,
                      do_constant_folding=True
                      )
    model_onnx = onnx.load(onnx_path)
    print(onnx.helper.printable_graph(model_onnx.graph))
    model_onnx, check = simplify(model_onnx, check_n=3)
    assert check, "Simplified ONNX model could not be validated"
    onnx.save(model_onnx, onnx_path)

    session = ort.InferenceSession(onnx_path)

    print(session.get_inputs())
    for ii in session.get_inputs():
        print("input: ", ii)
    for oo in session.get_outputs():
        print("output: ", oo)

    output11_ = session.run(None, input_feed={"input": x})[0]
    print(output11_-output_.detach().cpu().numpy())
    print("onnx output11_.shape: ", output11_.shape)

    output11_ = torch.from_numpy(output11_)
    batch_detections = non_max_suppression(output11_, len(class_names),
                                           conf_thres=confidence,
                                           nms_thres=iou)
    batch_detections = batch_detections[0].cpu().numpy()

    # ---------------------------------------------------------#
    #   对预测框进行得分筛选
    # ---------------------------------------------------------#
    top_index = batch_detections[:, 4] * batch_detections[:, 5] > confidence
    top_conf = batch_detections[top_index, 4] * batch_detections[top_index, 5]
    top_label = np.array(batch_detections[top_index, -1], np.int32)
    top_bboxes = np.array(batch_detections[top_index, :4])
    top_xmin, top_ymin, top_xmax, top_ymax = np.expand_dims(top_bboxes[:, 0], -1), np.expand_dims(
        top_bboxes[:, 1], -1), np.expand_dims(top_bboxes[:, 2], -1), np.expand_dims(top_bboxes[:, 3], -1)

    top_xmin = top_xmin / model_image_size[1] * image_shape[1]  # width
    top_ymin = top_ymin / model_image_size[0] * image_shape[0]
    top_xmax = top_xmax / model_image_size[1] * image_shape[1]
    top_ymax = top_ymax / model_image_size[0] * image_shape[0]
    boxes1 = np.concatenate([top_ymin, top_xmin, top_ymax, top_xmax], axis=-1)
    print(boxes1.shape)

    for i, c in enumerate(top_label):
        predicted_class = class_names[c]
        score = top_conf[i]

        top, left, bottom, right = boxes1[i]
        top = top - 5
        left = left - 5
        bottom = bottom + 5
        right = right + 5

        top = max(0, np.floor(top + 0.5).astype('int32'))
        left = max(0, np.floor(left + 0.5).astype('int32'))
        bottom = min(np.shape(img_copy)[0], np.floor(bottom + 0.5).astype('int32'))
        right = min(np.shape(img_copy)[1], np.floor(right + 0.5).astype('int32'))

        # 画框框
        label = '{} {:.2f}'.format(predicted_class, score)

        cv2.putText(img_copy, label, (left, top), 1, 1, (0, 255, 0), 1)
        cv2.rectangle(img_copy, (left, top), (right, bottom), (0, 0, 255), 1)

    cv2.imwrite("./img/street_onnx_v2.jpg", img_copy)
    print("Test onnx done!")
    print(f"Converted {model_path} to {onnx_path} done!")


if __name__ == "__main__":
    np.set_printoptions(suppress=True)
    convert_to_onnx(model_path="./model_data/yolov4_tiny_weights_coco.pth",
                    anchors_path="./model_data/yolo_anchors.txt",
                    classes_path="model_data/coco_classes.txt",
                    onnx_path="./model_data/yolov4_tiny_weights_coco.onnx",
                    phi=0)

    """
    PYTHONPATH=. python3 ./convert_to_onnx.py
    """
```  
