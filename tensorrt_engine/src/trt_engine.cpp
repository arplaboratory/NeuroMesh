#include "tensorrt_engine/trt_engine.h"

#include <iostream>
#include <memory>
#include <fstream>
#include <cassert>
#include <numeric>

#include "NvInferPlugin.h"


nvinfer1::ICudaEngine *createEngine(const std::string &inputFileName, nvinfer1::ILogger &logger) {

    std::string load_file;
    std::string engine_path = inputFileName; 
    //5 is length of ".onnx" and before ".trt"

    if(engine_path.find(".onnx") != std::string::npos){
        engine_path = engine_path.substr(0, engine_path.find(".onnx", engine_path.length() - 5)) + ".trt";
    }

    initLibNvInferPlugins(nullptr, "");

    //test if conversion already happened and .trt version exists
    std::ifstream f(engine_path.c_str());
    if(f.good()){ //".trt" file
        load_file = engine_path;

        std::vector<char> buffer;
        {
            std::ifstream in(engine_path, std::ios::binary | std::ios::ate);
            if (!in)
                throw std::runtime_error("Cannot open " + engine_path);
            std::streamsize ss = in.tellg();
            in.seekg(0, std::ios::beg);
            std::cout << "Input file size = " << ss << std::endl;
            buffer.resize(ss);
            if (0 == ss || !in.read(buffer.data(), ss))
                throw std::runtime_error("Cannot read" + engine_path + ".engine");
        }

        std::unique_ptr<nvinfer1::IRuntime> runtime(nvinfer1::createInferRuntime(logger));
        assert(runtime != nullptr);

        return runtime->deserializeCudaEngine(buffer.data(), buffer.size(), nullptr);

    }else{ //".onnx" file
        load_file = inputFileName;

        std::unique_ptr<nvinfer1::IBuilder> builder{nvinfer1::createInferBuilder(logger)};
        std::unique_ptr<nvinfer1::INetworkDefinition> network{
            builder->createNetworkV2(1U << (unsigned) nvinfer1::NetworkDefinitionCreationFlag::kEXPLICIT_BATCH)};

        std::unique_ptr<nvonnxparser::IParser> parser{ nvonnxparser::createParser(*network, logger) };

        if (!parser->parseFromFile(load_file.c_str(), static_cast<int>(nvinfer1::ILogger::Severity::kINFO)))
            throw std::runtime_error("ERROR: could not parse model file " + load_file + " !");

        std::unique_ptr<nvinfer1::IBuilderConfig> config(builder->createBuilderConfig());

        nvinfer1::ICudaEngine* engine = builder->buildEngineWithConfig(*network, *config);

        //save to .trt
        std::unique_ptr<nvinfer1::IHostMemory> serializedEngine(engine->serialize());

        std::ofstream out(engine_path, std::ios::binary);
        out.write((char *)serializedEngine->data(), serializedEngine->size());


        return engine;
    }
}

TRTEngine::TRTEngine(std::string model_filename, std::vector<uint> input_dims, int type_length, int batchSize){

    Logger logger;
    logger.log(nvinfer1::ILogger::Severity::kINFO, "Creating engine ...");

    engine.reset(createEngine(model_filename, logger));
    
    if (!engine)
        throw std::runtime_error("Engine creation failed !");

    logger.log(nvinfer1::ILogger::Severity::kINFO, "Engine created.");

    context.reset(engine->createExecutionContext());
    bindings.resize(engine->getNbBindings());

     // Alloc cuda memory for IO tensors
    for (int i = 0; i < engine->getNbBindings(); ++i) {
        nvinfer1::Dims dims{engine->getBindingDimensions(i)};
        size_t size = std::accumulate(dims.d, dims.d + dims.nbDims, batchSize, std::multiplies<size_t>());
        // Create CUDA buffer for Tensor.
        cudaMalloc(&bindings[i], size * type_length);
    }
}

TRTEngine::~TRTEngine(){

    for (int i = 0; i < bindings.size(); i++){
        cudaFree(bindings[i]);
    }
}

void TRTEngine::runInference( const void* inputTensor, int inputSize,  void* outputTensor, int outputSize){

    int inputId = 0;
    int outputId = 1; //only supporting 1 input 1 output

    cudaStream_t stream;
    cudaStreamCreate(&stream);

    cudaMemcpyAsync(bindings[inputId], inputTensor, inputSize, cudaMemcpyHostToDevice,
                    stream);
    context->enqueueV2(bindings.data(), stream, nullptr);
    cudaMemcpyAsync(outputTensor, bindings[outputId], outputSize,
                    cudaMemcpyDeviceToHost, stream);

    cudaStreamSynchronize(stream);
    cudaStreamDestroy(stream);
}
