//
// Created by wangzijian on 24-7-11.
//

#include "trt_handler.h"


using trtcore::BasicTRTHandler; // using namespace

BasicTRTHandler::BasicTRTHandler(const std::string &_trt_model_path, unsigned int _num_threads) : log_id(_trt_model_path.data()),num_threads(_num_threads){
    trt_model_path = _trt_model_path.data(); // model path
    initialize_handler();
    print_debug_string();
}

BasicTRTHandler::~BasicTRTHandler() {
    // don't need free by manunly
    cudaFree(buffers[0]);
    cudaFree(buffers[1]);
    cudaStreamDestroy(stream);
}


void BasicTRTHandler::initialize_handler() {
    // read engine file
    std::ifstream file(trt_model_path, std::ios::binary);

    if (!file.good()) {
        std::cerr << "Failed to read model file: " << trt_model_path << std::endl;
        return;
    }
    file.seekg(0, std::ifstream::end);
    size_t model_size = file.tellg();
    file.seekg(0, std::ifstream::beg);
    std::vector<char> model_data(model_size);
    file.read(model_data.data(), model_size);
    file.close();

    trt_runtime.reset(nvinfer1::createInferRuntime(trt_logger));
    // engine deserialize
    trt_engine.reset(trt_runtime->deserializeCudaEngine(model_data.data(), model_size));
    if (!trt_engine) {
        std::cerr << "Failed to deserialize the TensorRT engine." << std::endl;
        return;
    }
    trt_context.reset(trt_engine->createExecutionContext());
    if (!trt_context) {
        std::cerr << "Failed to create execution context." << std::endl;
        return;
    }
    cudaStreamCreate(&stream);

    nvinfer1::Dims input_dims = trt_engine->getTensorShape("images");
    nvinfer1::Dims output_dims = trt_engine->getTensorShape("output0");


    input_tensor_size = 1;
    for (int i = 0; i < input_dims.nbDims; ++i) {
        input_node_dims.push_back(input_dims.d[i]);
        input_tensor_size *= input_dims.d[i];
    }

    output_tensor_size = 1;
    for (int i = 0; i < output_dims.nbDims; ++i) {
        output_node_dims.push_back(output_dims.d[i]);
        output_tensor_size *= output_dims.d[i];
    }

    cudaMalloc(&buffers[0], input_tensor_size * sizeof(float));
    cudaMalloc(&buffers[1], output_tensor_size * sizeof(float));

    trt_context->setTensorAddress("images", buffers[0]);
    trt_context->setTensorAddress("output0", buffers[1]);

}

void BasicTRTHandler::print_debug_string() {
    std::cout << "TensorRT model loaded from: " << trt_model_path << std::endl;
    std::cout << "Input tensor size: " << input_tensor_size << std::endl;
    std::cout << "Output tensor size: " << output_tensor_size << std::endl;
}

