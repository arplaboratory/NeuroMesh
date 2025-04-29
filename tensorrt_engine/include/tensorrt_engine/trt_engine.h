#ifndef TRT_ENGINE_H
#define TRT_ENGINE_H

#include <vector>
#include <string>
#include <memory>

#include <NvInfer.h>
#include <NvOnnxParser.h>

#include <cuda_runtime.h>

#include <tensorrt_engine/logger.h>

class TRTEngine {
public:

    TRTEngine(std::string model_filename, std::vector<uint> input_dims, int type_length, int batchSize=1);
    ~TRTEngine();

    void runInference( const void* inputTensor, int inputSize, void* outputTensor, int outputSize);

protected:
    Logger logger;
    std::unique_ptr<nvinfer1::ICudaEngine> engine;

    std::unique_ptr<nvinfer1::IExecutionContext> context;
    std::vector<void*> bindings;

};


#endif