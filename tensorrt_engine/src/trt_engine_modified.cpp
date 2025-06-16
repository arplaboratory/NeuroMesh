#include "tensorrt_engine/trt_engine_modified.h"

#include <cassert>
#include <fstream>
#include <iostream>
#include <memory>
#include <numeric>

#include "NvInferPlugin.h"

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

TRTEngine::TRTEngine(std::string model_filename,
                     std::vector<std::vector<uint>> input_dims, int type_length,
                     int batchSize) {

  logger.log(nvinfer1::ILogger::Severity::kINFO, "Creating engine ...");

  runtime.reset(nvinfer1::createInferRuntime(logger));
  if (!runtime)
    throw std::runtime_error("Runtime creation failed!");

  engine.reset(createEngine(model_filename, logger, runtime));

  if (!engine)
    throw std::runtime_error("Engine creation failed !");

  logger.log(nvinfer1::ILogger::Severity::kINFO, "Engine created.");

  context.reset(engine->createExecutionContext());

  const int num_io = engine->getNbIOTensors();
  bindings.resize(num_io);

  // Alloc cuda memory for IO tensors
  for (int i = 0; i < num_io; ++i) {
    const char *tensor_name = engine->getIOTensorName(i);
    nvinfer1::Dims dims = engine->getTensorShape(tensor_name);
    size_t size = std::accumulate(dims.d, dims.d + dims.nbDims, 1,
                                  std::multiplies<size_t>());
    // Create CUDA buffer for Tensor
    cudaMalloc(&bindings[i], size * type_length);
  }
}

TRTEngine::~TRTEngine() {
  for (int i = 0; i < bindings.size(); i++) {
    cudaFree(bindings[i]);
  }
}

void TRTEngine::runInference(const std::vector<const void *> &inputTensors,
                             const std::vector<int> &inputSizes,
                             std::vector<void *> &outputTensors,
                             const std::vector<int> &outputSizes) {

  /*
  std::cout << "IN ENGINE: my inputTensors.size() looks like " << inputTensors.size() << std::endl; std::cout << "IN ENGINE: my inputSizes.size() looks like " << inputSizes.size() << std::endl;
  std::cout << "IN ENGINE: my inputSizes[0] looks like " << inputSizes.at(0) << std::endl;

  std::cout << "IN ENGINE: my outputTensors.size() looks like " << outputTensors.size() << std::endl; std::cout << "IN ENGINE: my outputSizes.size() looks like " << outputSizes.size() << std::endl;
  std::cout << "IN ENGINE: my outputSizes[0] looks like " << outputSizes.at(0) << std::endl;
  */

  if (inputTensors.size() + outputTensors.size() != bindings.size()) {
    throw std::runtime_error(
        "Number of input and output tensors doesn't match engine bindings");
  }

  cudaStream_t stream;

  cudaStreamCreate(&stream);

  // Copy input data to device
  for (size_t i = 0; i < inputTensors.size(); ++i) {
    cudaMemcpyAsync(bindings[i], inputTensors[i], inputSizes[i],
                    cudaMemcpyHostToDevice, stream);
  }

  for (size_t i = 0; i < inputTensors.size(); ++i) {
    const char *tensor_name = engine->getIOTensorName(i);
    context->setInputTensorAddress(tensor_name, bindings[i]);
  }

  for (size_t i = 0; i < outputTensors.size(); ++i) {
    const char *tensor_name = engine->getIOTensorName(inputTensors.size() + i);
    context->setOutputTensorAddress(tensor_name, bindings[inputTensors.size() + i]);
  }

  // Run inference
  context->enqueueV3(stream);

  // Copy output data to host
  for (size_t i = 0; i < outputTensors.size(); ++i) {
    cudaMemcpyAsync(outputTensors[i], bindings[inputTensors.size() + i],
                    outputSizes[i], cudaMemcpyDeviceToHost, stream);
  }

  cudaStreamSynchronize(stream);

  cudaStreamDestroy(stream);
}
