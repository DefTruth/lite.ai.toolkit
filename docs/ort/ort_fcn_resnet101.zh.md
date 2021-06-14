# How to convert FCNResNet101 to ONNX

## 1. 前言

这篇文档主要记录将项目[FCNResNet101](https://pytorch.org/hub/pytorch_vision_fcn_resnet101) 的模型转换成onnx模型.

## 2. 转换成ONNX模型

* 依赖库：
  * pytorch 1.8
  * onnx 1.7.0
  * onnxruntime 1.7.0
  * opencv 4.5.1
  * onnx-simplifier 0.3.5

```python
import torch
import cv2
import onnx
import onnxruntime as ort
from onnxsim import simplify

def convert_dynamic_fcn_onnx():
    import numpy as np
    onnx_path = "./weights/fcn_resnet101.onnx"
    sim_onnx_path = "./weights/fcn_resnet101-sim.onnx"
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    model = torch.hub.load('pytorch/vision:v0.9.0', 'fcn_resnet101', pretrained=True).to(device)
    model.eval()

    filename = "./data/dog.jpg"

    input_image = cv2.imread(filename)
    input_image = cv2.resize(input_image, (512, 512))
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
                      output_names=['out'],  # the model's output names
                      dynamic_axes={'input': {2: 'height', 3: 'width'},
                                    'out': {2: 'height', 3: 'width'}}
                      )
    print("export onnx done.")
    onnx_model = onnx.load(onnx_path)
    onnx.checker.check_model(onnx_model)
    print(onnx.helper.printable_graph(onnx_model.graph))

    model_simp, check = simplify(onnx_model,
                                 check_n=3,
                                 dynamic_input_shape=True,
                                 input_shapes={"input": [1, 3, 512, 512]}
                                 )
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

    out = ort_session.run(['out'], input_feed={"input": x_numpy})[0]

    print("deeplabv3 out.shape: ", out.shape)

    output_predictions = out[0].argmax(0)
    print(output_predictions.shape)  # (h,w)
    print(output_predictions.min())
    print(output_predictions.max())
    mask0 = output_predictions * 10  # 不同的类别 选择不同的颜色
    mask1 = (output_predictions - 21) * 5 + 210   # 不同的类别 选择不同的颜色
    mask2 = (output_predictions - 10) * 5 + 110   # 不同的类别 选择不同的颜色
    mask0 = mask0.astype(np.uint8)
    mask1 = mask1.astype(np.uint8)
    mask2 = mask2.astype(np.uint8)
    mask0 = np.expand_dims(mask0, 2)
    mask1 = np.expand_dims(mask1, 2)
    mask2 = np.expand_dims(mask2, 2)
    mask = np.concatenate([mask0, mask1, mask2], 2)
    mask = mask * np.expand_dims(output_predictions, 2)
    mask = mask.astype(np.uint8)
    cv2.imwrite("./data/dog_mask.jpg", mask)

    output_predictions = torch.from_numpy(output_predictions)

    # create a color pallette, selecting a color for each class
    palette = torch.tensor([2 ** 25 - 1, 2 ** 15 - 1, 2 ** 21 - 1])
    colors = torch.as_tensor([i for i in range(21)])[:, None] * palette
    colors = (colors % 255).numpy().astype("uint8")

    # plot the semantic segmentation predictions of 21 classes in each color
    from PIL import Image
    r = Image.fromarray(output_predictions.byte().cpu().numpy()).resize((512, 512))
    r.putpalette(colors)

    import matplotlib.pyplot as plt
    plt.imshow(r)
    plt.savefig("./data/dog_onnx.jpg")


if __name__ == "__main__":
    convert_dynamic_fcn_onnx()

    """
    PYTHONPATH=. python3 ./fcn_resnet101.py
    """
```

