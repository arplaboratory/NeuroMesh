#include "tensorrt_engine/trt_engine_modified.h"

#include <cassert>
#include <fstream>
#include <iostream>
#include <memory>
#include <numeric>

#include <NvInfer.h>
#include <NvOnnxParser.h>
#include "NvInferPlugin.h"
#include <cuda_runtime.h>

namespace engine_interface
{

class TRTEngine::Impl
{
public:
  Logger logger;
  std::unique_ptr<nvinfer1::ICudaEngine> engine;
  std::unique_ptr<nvinfer1::IRuntime> runtime;
  std::unique_ptr<nvinfer1::IExecutionContext> context;
  std::vector<void*> bindings;

  Impl() : logger(), engine(nullptr), runtime(nullptr), context(nullptr) {
    // Initialize the logger
    logger.log(nvinfer1::ILogger::Severity::kINFO, "TRTEngine Impl initialized.");
  }
};

TRTEngine::TRTEngine()
    : impl_(std::make_unique<Impl>()) {
}

TRTEngine::~TRTEngine() {
  for (size_t i = 0; i < impl_->bindings.size(); i++) {
    cudaFree(impl_->bindings[i]);
  }
}

nvinfer1::ICudaEngine *createEngine(const std::string &inputFileName,
                                    nvinfer1::ILogger &logger,
                                    std::unique_ptr<nvinfer1::IRuntime> &runtime) {
  std::string load_file;
  std::string engine_path = inputFileName;
  // 5 is length of ".onnx" and before ".trt"

  if (engine_path.find(".onnx") != std::string::npos) {
    engine_path = engine_path.substr(
                      0, engine_path.find(".onnx", engine_path.length() - 5)) +
                  ".trt";
  }

  initLibNvInferPlugins(nullptr, "");

  // test if conversion already happened and .trt version exists
  std::ifstream f(engine_path.c_str());
  if (f.good()) { //".trt" file
    load_file = engine_path;

    std::vector<char> buffer;
    {
      std::ifstream in(engine_path, std::ios::binary | std::ios::ate);
      if (!in)
        throw std::runtime_error("Cannot open " + engine_path);
      std::streamsize ss = in.tellg();
      in.seekg(0, std::ios::beg);
      // std::cout << "Input file size = " << ss << std::endl;
      buffer.resize(ss);
      if (0 == ss || !in.read(buffer.data(), ss))
        throw std::runtime_error("Cannot read" + engine_path + ".engine");
    }

    return runtime->deserializeCudaEngine(buffer.data(), buffer.size());

  } else { //".onnx" file
    load_file = inputFileName;

    std::unique_ptr<nvinfer1::IBuilder> builder{
        nvinfer1::createInferBuilder(logger)};
    std::unique_ptr<nvinfer1::INetworkDefinition> network{
        builder->createNetworkV2(
            1U << (unsigned)
                nvinfer1::NetworkDefinitionCreationFlag::kEXPLICIT_BATCH)};

    std::unique_ptr<nvonnxparser::IParser> parser{
        nvonnxparser::createParser(*network, logger)};

    if (!parser->parseFromFile(
            load_file.c_str(),
            static_cast<int>(nvinfer1::ILogger::Severity::kINFO)))
      throw std::runtime_error("ERROR: could not parse model file " +
                               load_file + " !");

    std::unique_ptr<nvinfer1::IBuilderConfig> config(
        builder->createBuilderConfig());

    std::unique_ptr<nvinfer1::IHostMemory> serializedEngine(
        builder->buildSerializedNetwork(*network, *config));

    // save to .trt
    std::ofstream out(engine_path, std::ios::binary);
    out.write((char *)serializedEngine->data(), serializedEngine->size());

    // deserialize from serialized engine
    std::unique_ptr<nvinfer1::IRuntime> runtime(
        nvinfer1::createInferRuntime(logger));
    assert(runtime != nullptr);

    return runtime->deserializeCudaEngine(serializedEngine->data(),
                                          serializedEngine->size());
  }
}

bool TRTEngine::loadModel(const std::string& model_path,
                     const std::vector<std::vector<uint32_t>>& input_dims, 
                     int type_length) 
{
  impl_->logger.log(nvinfer1::ILogger::Severity::kINFO, "Creating engine ...");

  impl_->runtime.reset(nvinfer1::createInferRuntime(impl_->logger));
  if (!impl_->runtime)
    throw std::runtime_error("Runtime creation failed!");

  impl_->engine.reset(createEngine(model_path, impl_->logger, impl_->runtime));

  if (!impl_->engine)
    throw std::runtime_error("Engine creation failed !");

  impl_->logger.log(nvinfer1::ILogger::Severity::kINFO, "Engine created.");

  impl_->context.reset(impl_->engine->createExecutionContext());

  const int num_io = impl_->engine->getNbIOTensors();
  impl_->bindings.resize(num_io);

  // Alloc cuda memory for IO tensors
  for (int i = 0; i < num_io; ++i) {
    const char *tensor_name = impl_->engine->getIOTensorName(i);
    nvinfer1::Dims dims = impl_->engine->getTensorShape(tensor_name);
    size_t size = std::accumulate(dims.d, dims.d + dims.nbDims, 1,
                                  std::multiplies<size_t>());
    // Create CUDA buffer for Tensor
    cudaMalloc(&impl_->bindings[i], size * type_length);
  }

  return true;
}

void TRTEngine::runInference(const std::vector<const void *> &inputTensors,
                             const std::vector<int> &inputSizes,
                             std::vector<void *> &outputTensors,
                             const std::vector<int> &outputSizes) {

  if (inputTensors.size() + outputTensors.size() != impl_->bindings.size()) {
    throw std::runtime_error(
        "Number of input and output tensors doesn't match engine bindings");
  }

  cudaStream_t stream;

  cudaStreamCreate(&stream);

  // Copy input data to device
  for (size_t i = 0; i < inputTensors.size(); ++i) {
    cudaMemcpyAsync(impl_->bindings[i], inputTensors[i], inputSizes[i],
                    cudaMemcpyHostToDevice, stream);
  }

  for (size_t i = 0; i < inputTensors.size(); ++i) {
    const char *tensor_name = impl_->engine->getIOTensorName(i);
    impl_->context->setInputTensorAddress(tensor_name, impl_->bindings[i]);
  }

  for (size_t i = 0; i < outputTensors.size(); ++i) {
    const char *tensor_name = impl_->engine->getIOTensorName(inputTensors.size() + i);
    impl_->context->setOutputTensorAddress(tensor_name, impl_->bindings[inputTensors.size() + i]);
  }

  // Run inference
  impl_->context->enqueueV3(stream);

  // Copy output data to host
  for (size_t i = 0; i < outputTensors.size(); ++i) {
    cudaMemcpyAsync(outputTensors[i], impl_->bindings[inputTensors.size() + i],
                    outputSizes[i], cudaMemcpyDeviceToHost, stream);
  }

  cudaStreamSynchronize(stream);

  cudaStreamDestroy(stream);
}
} // namespace engine_interface