# How to convert MobileNetV2 to ONNX

## 1. 前言

这篇文档主要记录将项目[MobileNetV2](https://pytorch.org/hub/pytorch_vision_mobilenet_v2) 的模型转换成onnx模型.

## 2. 转换成ONNX模型

* 依赖库：
  * pytorch 1.8
  * onnx 1.7.0
  * onnxruntime 1.7.0
  * opencv 4.5.1
  * onnx-simplifier 0.3.5

```python
import torch
from PIL import Image
from torchvision import transforms
import cv2
import onnx
import onnxruntime as ort
from onnxsim import simplify


def convert_static_mobilenetv2():
    import numpy as np

    onnx_path = "./weights/mobilenetv2.onnx"
    sim_onnx_path = "./weights/mobilenetv2-sim.onnx"
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    model = torch.hub.load('pytorch/vision:v0.9.0', 'mobilenet_v2', pretrained=True).to(device)

    model.eval()

    filename = "./data/dog.jpg"

    input_image = cv2.imread(filename)
    input_image = cv2.resize(input_image, (224, 224))
    input_image = cv2.cvtColor(input_image, cv2.COLOR_BGR2RGB)

    input_image = input_image.astype(np.float32) / 255.
    input_image[:, :, 0] -= 0.485
    input_image[:, :, 1] -= 0.456
    input_image[:, :, 2] -= 0.406
    input_image[:, :, 0] /= 0.229
    input_image[:, :, 1] /= 0.224
    input_image[:, :, 2] /= 0.225

    input_image = input_image.transpose((2, 0, 1))

    input_tensor = torch.from_numpy(input_image)  # (3,512,512)
    print(input_tensor.shape)
    x = input_tensor.unsqueeze(0).cpu()  # create a mini-batch as expected by the model

    print("converting onnx ...")
    torch.onnx.export(model,  # model being run
                      x,  # model input (or a tuple for multiple inputs)
                      onnx_path,  # where to save the model (can be a file or file-like object)
                      export_params=True,  # store the trained parameter weights inside the model file
                      opset_version=11,  # the ONNX version to export the model to
                      do_constant_folding=True,  # whether to execute constant folding for optimization
                      input_names=['input'],  # the model's input names
                      output_names=['logits'],  # the model's output names
                      )
    print("export onnx done.")
    onnx_model = onnx.load(onnx_path)
    onnx.checker.check_model(onnx_model)
    print(onnx.helper.printable_graph(onnx_model.graph))

    model_simp, check = simplify(onnx_model, check_n=3)
    onnx.save(model_simp, sim_onnx_path)
    print(onnx.helper.printable_graph(model_simp.graph))
    print("export onnx sim done.")

    # test onnxruntime
    ort_session = ort.InferenceSession(sim_onnx_path)

    for o in ort_session.get_inputs():
        print(o)

    for o in ort_session.get_outputs():
        print(o)

    x_numpy = x.cpu().numpy()

    logits = ort_session.run(['logits'], input_feed={"input": x_numpy})[0]  # (1,1000)
    print(logits.shape)
    logits = torch.from_numpy(logits)
    probabilities = torch.softmax(logits[0], dim=0)  # (1000,)
    print(probabilities.shape)

    # Read the categories
    with open("./data/imagenet_classes.txt", "r") as f:
        categories = [s.strip() for s in f.readlines()]
    # Show top categories per image
    top5_prob, top5_catid = torch.topk(probabilities, 5)
    for i in range(top5_prob.size(0)):
        print(categories[top5_catid[i]], top5_prob[i].item())


if __name__ == "__main__":
    convert_static_mobilenetv2()

    """
    PYTHONPATH=. python3 ./mobilenetv2.py
    """

```

