//
// Created by ai-test1 on 24-7-11.
//

#ifndef LITE_AI_TOOLKIT_LOGGER_H
#define LITE_AI_TOOLKIT_LOGGER_H
#ifndef LOGGER_H
#define LOGGER_H

#include "NvInfer.h"
#include <iostream>

class Logger : public nvinfer1::ILogger {
public:
    void log(Severity severity, const char* msg) noexcept override {
        if (severity != Severity::kINFO) {
            std::cout << "TensorRT: " << msg << std::endl;
        }
    }
};

#endif // LOGGER_H
#endif //LITE_AI_TOOLKIT_LOGGER_H
