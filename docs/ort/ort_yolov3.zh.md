# ONNXRuntime inference for YOLOV3 and Tiny-YOLOV3

## 1. 前言

这篇文档主要记录将项目[YOLOV3](https://github.com/onnx/models/blob/master/vision/object_detection_segmentation/yolov3)  以及 [Tiny-YOLOV3](https://github.com/onnx/models/blob/master/vision/object_detection_segmentation/tiny-yolov3) 的python版本onnxruntime的推理重写。

## 2. 推理代码

```python
import numpy as np
from PIL import Image
import onnxruntime as ort
import torchvision
import cv2

class_names = ['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light',
               'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
               'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
               'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard',
               'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
               'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
               'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard',
               'cell phone',
               'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors',
               'teddy bear',
               'hair drier', 'toothbrush']


# this function is from yolo3.utils.letterbox_image
def letterbox_image(image, size):
    '''resize image with unchanged aspect ratio using padding'''
    iw, ih = image.size
    w, h = size
    scale = min(w / iw, h / ih)
    nw = int(iw * scale)
    nh = int(ih * scale)

    image = image.resize((nw, nh), Image.BICUBIC)
    new_image = Image.new('RGB', size, (128, 128, 128))
    new_image.paste(image, ((w - nw) // 2, (h - nh) // 2))
    return new_image


def letterbox_image_cv2(img_bgr: np.ndarray, size: tuple = (416, 416)):
    '''resize image with unchanged aspect ratio using padding'''
    iw, ih = img_bgr.shape[1], img_bgr.shape[0]
    w, h = size
    scale = min(w / iw, h / ih)
    nw = int(iw * scale)
    nh = int(ih * scale)

    image = cv2.resize(img_bgr, (nw, nh))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # image = image.resize((nw, nh), Image.BICUBIC)
    # new_image = Image.new('RGB', size, (128, 128, 128))
    new_image = np.zeros((h, w, 3), dtype=np.uint8)  # 对应cv::Mat mat(h,w,CV_8UC3,128)
    new_image.fill(128)
    # 对应 cv::Mat::convertTo(mat(roi))接口 roi是cv::Rect
    new_image[(h - nh) // 2:(h - nh) // 2 + nh, (w - nw) // 2:(w - nw) // 2 + nw, :] = image[:, :, :]
    # new_image.paste(image, ((w - nw) // 2, (h - nh) // 2))
    return new_image  # RGB


def preprocess_cv2(img):
    model_image_size = (512, 512)
    boxed_image = letterbox_image_cv2(img, tuple(reversed(model_image_size)))
    image_data = np.array(boxed_image, dtype='float32')
    image_data /= 255.
    image_data = np.transpose(image_data, [2, 0, 1])
    image_data = np.expand_dims(image_data, 0)
    return image_data


def preprocess(img):
    model_image_size = (416, 416)
    boxed_image = letterbox_image(img, tuple(reversed(model_image_size)))
    image_data = np.array(boxed_image, dtype='float32')
    image_data /= 255.
    image_data = np.transpose(image_data, [2, 0, 1])
    image_data = np.expand_dims(image_data, 0)
    return image_data


def infer_yolov3():
    onnx_path = "./yolov3-10.onnx"
    ort.set_default_logger_severity(4)
    ort_session = ort.InferenceSession(onnx_path)

    img_path = "./bus.jpg"
    save_path = "./bus_yolov3.jpg"
    image = Image.open(img_path)
    # input
    image_data = preprocess(image)
    image_size = np.array([image.size[1], image.size[0]], dtype=np.float32).reshape(1, 2)
    # input

    input_name = ort_session.get_inputs()[0].name
    size_name = ort_session.get_inputs()[1].name
    print(input_name, size_name)

    # boxes: (1x'n_candidates'x4) the coordinates of all anchor boxes (y1,x1,y2,x2)
    # scores: (1x80x'n_candidates') the scores of all anchor boxes per class
    # indices: ('nbox'x3) selected indices from the boxes tensor.
    # The selected index format is (batch_index, class_index, box_index)
    boxes, scores, indices = ort_session.run(None, input_feed={input_name: image_data,
                                                               size_name: image_size})

    out_boxes, out_scores, out_classes = [], [], []
    for idx_ in indices:
        # (n,3) -> (batch_index, class_index, box_index)
        out_classes.append(idx_[1].item())
        out_scores.append(scores[tuple(idx_)].item())
        idx_1 = (idx_[0], idx_[2])
        out_boxes.append(boxes[idx_1].flatten().tolist())

    print(f"detect {len(out_boxes)} bounding boxes.")
    print(out_boxes)
    print(out_scores)
    print(out_classes)

    img = cv2.imread(img_path)
    print("cv2 img shape: ", img.shape)
    print("PIL img shape: ", image.size)

    for i in range(len(out_boxes)):
        y1, x1, y2, x2 = out_boxes[i]
        conf, label = out_scores[i], out_classes[i]
        x1, y1, x2, y2, label = int(x1), int(y1), int(x2), int(y2), int(label)
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2, 2)
        cv2.putText(img, class_names[label] + ":{:.2f}".format(conf), (x1, y1),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2, 2)

    cv2.imwrite(save_path, img)

    print("detect done.")


def infer_yolov3_cv2():
    onnx_path = "./tiny-yolov3-11.onnx"
    ort.set_default_logger_severity(4)
    ort_session = ort.InferenceSession(onnx_path)

    img_path = "./bus.jpg"
    save_path = "./bus_yolov3_cv2.jpg"
    image = cv2.imread(img_path)  # bgr
    height, width, _ = image.shape
    # input
    image_data = preprocess_cv2(image)
    image_size = np.array([height, width], dtype=np.float32).reshape(1, 2)
    # input

    input_name = ort_session.get_inputs()[0].name
    size_name = ort_session.get_inputs()[1].name
    print(input_name, size_name)

    # boxes: (1x'n_candidates'x4) the coordinates of all anchor boxes (y1,x1,y2,x2)
    # scores: (1x80x'n_candidates') the scores of all anchor boxes per class
    # indices: ('nbox'x3) selected indices from the boxes tensor.
    # The selected index format is (batch_index, class_index, box_index)
    boxes, scores, indices = ort_session.run(None, input_feed={input_name: image_data,
                                                               size_name: image_size})

    print(boxes.shape)
    print(scores.shape)
    print(indices.shape)

    out_boxes, out_scores, out_classes = [], [], []
    for idx_ in indices[0]:
        # (n,3) -> (batch_index, class_index, box_index)
        out_classes.append(idx_[1].item())
        out_scores.append(scores[tuple(idx_)].item())
        idx_1 = (idx_[0], idx_[2])
        out_boxes.append(boxes[idx_1].flatten().tolist())

    print(f"detect {len(out_boxes)} bounding boxes.")
    print(out_boxes)
    print(out_scores)
    print(out_classes)

    print("cv2 img shape: ", image.shape)

    for i in range(len(out_boxes)):
        y1, x1, y2, x2 = out_boxes[i]
        conf, label = out_scores[i], out_classes[i]
        x1, y1, x2, y2, label = int(x1), int(y1), int(x2), int(y2), int(label)
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2, 2)
        cv2.putText(image, class_names[label] + ":{:.2f}".format(conf), (x1, y1),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2, 2)

    cv2.imwrite(save_path, image)

    input_infos = ort_session.get_inputs()

    print(input_infos[0])
    print(input_infos[1])

    print("detect done.")


if __name__ == "__main__":
    infer_yolov3_cv2()
    """
        PYTHONPATH=. python3 ./yolov3_onnx_infer.py
    """

```



