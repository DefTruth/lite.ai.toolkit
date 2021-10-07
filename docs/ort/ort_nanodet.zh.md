
```shell
➜  weights git:(main) ✗ onnx2ncnn nanodet_m_1.5x_r.onnx nanodet_m_1.5x.param nanodet_m_1.5x.bin
➜  weights git:(main) ✗ onnx2ncnn nanodet_m_1.5x_416_r.onnx nanodet_m_1.5x_416.param nanodet_m_1.5x_416.bin
➜  weights git:(main) ✗ onnx2ncnn nanodet_m_416_r.onnx nanodet_m_416.param nanodet_m_416.bin 
➜  weights git:(main) ✗ onnx2ncnn nanodet_g_r.onnx nanodet_g.param nanodet_g.bin
➜  weights git:(main) ✗ onnx2ncnn nanodet_t_r.onnx nanodet_t.param nanodet_t.bin
➜  weights git:(main) ✗ onnx2ncnn nanodet-RepVGG-A0_416_r.onnx nanodet-RepVGG-A0_416.param nanodet-RepVGG-A0_416.bin
➜  weights git:(main) ✗ onnx2ncnn nanodet-EfficientNet-Lite0_320_r.onnx nanodet-EfficientNet-Lite0_320.param nanodet-EfficientNet-Lite0_320.bin
➜  weights git:(main) ✗ onnx2ncnn nanodet-EfficientNet-Lite1_416_r.onnx nanodet-EfficientNet-Lite1_416.param nanodet-EfficientNet-Lite1_416.bin
➜  weights git:(main) ✗ onnx2ncnn nanodet-EfficientNet-Lite2_512_r.onnx nanodet-EfficientNet-Lite2_512.param nanodet-EfficientNet-Lite2_512.bin

```  

```shell
➜  weights git:(main) ✗ ncnnoptimize nanodet_m.param nanodet_m.bin nanodet_m-opt.param nanodet_m-opt.bin 0
fuse_convolution_activation Conv_0 LeakyRelu_1
fuse_convolution_activation Conv_4 LeakyRelu_5
fuse_convolution_activation Conv_6 LeakyRelu_7
fuse_convolution_activation Conv_9 LeakyRelu_10
fuse_convolution_activation Conv_31 LeakyRelu_32
fuse_convolution_activation Conv_34 LeakyRelu_35
fuse_convolution_activation Conv_56 LeakyRelu_57
fuse_convolution_activation Conv_59 LeakyRelu_60
fuse_convolution_activation Conv_81 LeakyRelu_82
fuse_convolution_activation Conv_84 LeakyRelu_85
fuse_convolution_activation Conv_93 LeakyRelu_94
fuse_convolution_activation Conv_95 LeakyRelu_96
fuse_convolution_activation Conv_98 LeakyRelu_99
fuse_convolution_activation Conv_120 LeakyRelu_121
fuse_convolution_activation Conv_123 LeakyRelu_124
fuse_convolution_activation Conv_145 LeakyRelu_146
fuse_convolution_activation Conv_148 LeakyRelu_149
fuse_convolution_activation Conv_170 LeakyRelu_171
fuse_convolution_activation Conv_173 LeakyRelu_174
fuse_convolution_activation Conv_195 LeakyRelu_196
fuse_convolution_activation Conv_198 LeakyRelu_199
fuse_convolution_activation Conv_220 LeakyRelu_221
fuse_convolution_activation Conv_223 LeakyRelu_224
fuse_convolution_activation Conv_245 LeakyRelu_246
fuse_convolution_activation Conv_248 LeakyRelu_249
fuse_convolution_activation Conv_270 LeakyRelu_271
fuse_convolution_activation Conv_273 LeakyRelu_274
fuse_convolution_activation Conv_282 LeakyRelu_283
fuse_convolution_activation Conv_284 LeakyRelu_285
fuse_convolution_activation Conv_287 LeakyRelu_288
fuse_convolution_activation Conv_309 LeakyRelu_310
fuse_convolution_activation Conv_312 LeakyRelu_313
fuse_convolution_activation Conv_334 LeakyRelu_335
fuse_convolution_activation Conv_337 LeakyRelu_338
fuse_convolution_activation Conv_359 LeakyRelu_360
fuse_convolution_activation Conv_362 LeakyRelu_363
fuse_convolution_activation Conv_387 LeakyRelu_388
fuse_convolution_activation Conv_391 LeakyRelu_392
fuse_convolution_activation Conv_412 LeakyRelu_413
fuse_convolution_activation Conv_416 LeakyRelu_417
fuse_convolution_activation Conv_437 LeakyRelu_438
fuse_convolution_activation Conv_441 LeakyRelu_442
fuse_convolutiondepthwise_activation Conv_385 LeakyRelu_386
fuse_convolutiondepthwise_activation Conv_389 LeakyRelu_390
fuse_convolutiondepthwise_activation Conv_410 LeakyRelu_411
fuse_convolutiondepthwise_activation Conv_414 LeakyRelu_415
fuse_convolutiondepthwise_activation Conv_435 LeakyRelu_436
fuse_convolutiondepthwise_activation Conv_439 LeakyRelu_440
Input layer input.1 without shape info, shape_inference skipped
Input layer input.1 without shape info, estimate_memory_footprint skipped

```