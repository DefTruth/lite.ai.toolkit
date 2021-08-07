# 记录YoloR工程化

## 项目地址

* [yolor](https://github.com/WongKinYiu/yolor)  

## 转换成ONNX的过程

You must convert the saved SyncBatchNorm in pretrained yolor-xx.pt to BatchNorm first ! if you reload it use CPU.
```python  
import argparse
import sys
import time

sys.path.append('./')  # to run '$ python *.py' files in subdirectories

import torch
import torch.nn as nn
import models
from models.experimental import attempt_load
from utils.activations import Hardswish
from utils.general import set_logging, check_img_size


# convert SyncBatchNorm to BatchNorm2d
def convert_sync_batchnorm_to_batchnorm(module):
    module_output = module
    if isinstance(module, torch.nn.modules.batchnorm.SyncBatchNorm):
        module_output = torch.nn.BatchNorm2d(module.num_features,
                                             module.eps, module.momentum,
                                             module.affine,
                                             module.track_running_stats)

        if module.affine:
            with torch.no_grad():
                module_output.weight = module.weight
                module_output.bias = module.bias
        module_output.running_mean = module.running_mean
        module_output.running_var = module.running_var
        module_output.num_batches_tracked = module.num_batches_tracked
        if hasattr(module, "qconfig"):
            module_output.qconfig = module.qconfig
    for name, child in module.named_children():
        module_output.add_module(name, convert_sync_batchnorm_to_batchnorm(child))
    del module
    return module_output


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', type=str, default='./yolor-p6.pt', help='weights path')
    parser.add_argument('--img-size', nargs='+', type=int, default=[1280, 1280], help='image size')  # height, width
    parser.add_argument('--batch-size', type=int, default=1, help='batch size')
    opt = parser.parse_args()
    opt.img_size *= 2 if len(opt.img_size) == 1 else 1  # expand
    print(opt)
    set_logging()
    t = time.time()

    # Load PyTorch model
    model = attempt_load(opt.weights, map_location=torch.device('cpu'))  # load FP32 model
    labels = model.names
    model.eval()
    model = model.to("cpu")

    model = convert_sync_batchnorm_to_batchnorm(model)

    print(model)

    # Checks
    gs = int(max(model.stride))  # grid size (max stride)
    opt.img_size = [check_img_size(x, gs) for x in opt.img_size]  # verify img_size are gs-multiples

    # Input
    img = torch.zeros(opt.batch_size, 3, *opt.img_size)  # image size(1,3,320,192) iDetection

    # Update model
    for k, m in model.named_modules():
        m._non_persistent_buffers_set = set()  # pytorch 1.6.0 compatibility
        if isinstance(m, models.common.Conv) and isinstance(m.act, nn.Hardswish):
            m.act = Hardswish()  # assign activation
        # if isinstance(m, models.yolo.Detect):
        #     m.forward = m.forward_export  # assign forward (optional)
    model.model[-1].export = True  # set Detect() layer export=True
    y = model(img)  # dry run

    # TorchScript export
    # try:
    #     print('\nStarting TorchScript export with torch %s...' % torch.__version__)
    #     f = opt.weights.replace('.pt', '.torchscript.pt')  # filename
    #     ts = torch.jit.trace(model, img)
    #     ts.save(f)
    #     print('TorchScript export success, saved as %s' % f)
    # except Exception as e:
    #     print('TorchScript export failure: %s' % e)

    # ONNX export
    try:
        import onnx
        import onnxruntime as ort

        print('\nStarting ONNX export with onnx %s...' % onnx.__version__)
        f = opt.weights.replace('.pt', f'-{opt.img_size[0]}-{opt.img_size[1]}.onnx')  # filename
        torch.onnx.export(model, img, f, verbose=False, opset_version=12, input_names=['images'],
                          output_names=['classes', 'boxes'] if y is None else ['output'])

        # Checks
        onnx_model = onnx.load(f)  # load onnx model
        onnx.checker.check_model(onnx_model)  # check onnx model
        print(onnx.helper.printable_graph(onnx_model.graph))  # print a human readable model

        do_simplify = True
        if do_simplify:
            from onnxsim import simplify

            onnx_model, check = simplify(onnx_model, check_n=3)
            assert check, 'assert simplify check failed'
            onnx.save(onnx_model, f)

        session = ort.InferenceSession(f)

        for ii in session.get_inputs():
            print("input: ", ii)

        for oo in session.get_outputs():
            print("output: ", oo)

        print('ONNX export success, saved as %s' % f)
    except Exception as e:
        print('ONNX export failure: %s' % e)

    # CoreML export
    # try:
    #     import coremltools as ct
    #
    #     print('\nStarting CoreML export with coremltools %s...' % ct.__version__)
    #     # convert model from torchscript and apply pixel scaling as per detect.py
    #     model = ct.convert(ts, inputs=[ct.ImageType(name='image', shape=img.shape, scale=1 / 255.0, bias=[0, 0, 0])])
    #     f = opt.weights.replace('.pt', '.mlmodel')  # filename
    #     model.save(f)
    #     print('CoreML export success, saved as %s' % f)
    # except Exception as e:
    #     print('CoreML export failure: %s' % e)

    # Finish
    print('\nExport complete (%.2fs). Visualize with https://github.com/lutzroeder/netron.' % (time.time() - t))

    """
    PYTHONPATH=. python3 ./models/export.py --weights ./weights/yolor-p6.pt --img-size 640 
    """

```  
also there is a bug in the source code will make the converted onnx get wrong output shape.
```python
class IDetect(nn.Module):
   # ...
    def forward(self, x):
        # x = x.copy()  # for profiling
        z = []  # inference output
        # self.training |= self.export # 这句有点问题 error 
        self.training = not self.export  # change to this line
        print("self.training: ", self.training, self.nl)
```  
my log:
```python
Checking 0/3...
Checking 1/3...
Checking 2/3...
input:  NodeArg(name='images', type='tensor(float)', shape=[1, 3, 320, 320])
output:  NodeArg(name='output', type='tensor(float)', shape=[1, 6375, 85])
output:  NodeArg(name='910', type='tensor(float)', shape=[1, 3, 40, 40, 85])
output:  NodeArg(name='944', type='tensor(float)', shape=[1, 3, 20, 20, 85])
output:  NodeArg(name='978', type='tensor(float)', shape=[1, 3, 10, 10, 85])
output:  NodeArg(name='1012', type='tensor(float)', shape=[1, 3, 5, 5, 85])
ONNX export success, saved as ./weights/yolor-p6-320-320.onnx

```  
还有一段逻辑在导出成onnx的时候有问题，直到我修改成和yolov5一样之后，导出的onnx就可以跑出正常的结果了. 也许是pytorch的自动broadcast没有在onnx中执行
```python
            if not self.training:  # inference
                # if self.grid[i].shape[2:4] != x[i].shape[2:4]:
                #     self.grid[i] = self._make_grid(nx, ny).to(x[i].device)

                self.grid[i] = self._make_grid(nx, ny).to(x[i].device)
                y = x[i].sigmoid()
                # in yolor
                # y[..., 0:2] = (y[..., 0:2] * 2. - 0.5 + self.grid[i]) * self.stride[i]  # xy
                # y[..., 2:4] = (y[..., 2:4] * 2) ** 2 * self.anchor_grid[i]  # wh
                
                # change it as yolov5
                xy = (y[..., 0:2] * 2. - 0.5 + self.grid[i]) * self.stride[i]  # xy (bs,na,ny,nx,2)
                wh = (y[..., 2:4] * 2) ** 2 * self.anchor_grid[i].view(1, self.na, 1, 1, 2)  # wh (bs,na,ny,nx,2)
                y = torch.cat((xy, wh, y[..., 4:]), -1)  # (bs,na,ny,nx,2+2+1+nc=xy+wh+conf+cls_prob)

                z.append(y.view(bs, -1, self.no))
```  
在pytorch中的直接内存操作，在转换onnx后会有问题。似乎onnx还不支持这种操作，虽然能转换，但是效果无法对齐。因此将源码的直接内存操作修改成普通操作，就没有问题了。

