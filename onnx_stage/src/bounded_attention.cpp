// CUDA-only scheduling adapter. The arithmetic remains ORT's own MatMul and
// Softmax kernels; only independent leading-axis patches are evaluated in batches.
#define ORT_API_MANUAL_INIT
#include <onnxruntime_cxx_api.h>
#include <cuda_runtime_api.h>
#include <algorithm>
#include <array>
#include <mutex>
#include <stdexcept>

namespace {
void cuda_check(cudaError_t status) {
  if (status != cudaSuccess) throw std::runtime_error(cudaGetErrorString(status));
}
struct Scratch {
  void* pointer{};
  cudaStream_t stream;
  Scratch(std::size_t bytes,cudaStream_t s):stream(s) { cuda_check(cudaMallocAsync(&pointer,bytes,stream)); }
  ~Scratch() { if(pointer) cudaFreeAsync(pointer,stream); }
  Scratch(const Scratch&)=delete;
  Scratch& operator=(const Scratch&)=delete;
};
struct Kernel {
  Ort::Op matmul{nullptr}, softmax{nullptr};
  std::int64_t patch_batch;
  explicit Kernel(const OrtKernelInfo* info) {
    patch_batch=Ort::ConstKernelInfo(info).GetAttribute<std::int64_t>("patch_batch");
    if(patch_batch<1 || patch_batch>4) throw std::invalid_argument("attention patch_batch must be 1..4");
    const char* constraints[]{"T"};
    const ONNXTensorElementDataType types[]{ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT};
    matmul=Ort::Op::Create(info,"MatMul","",13,constraints,types,1,nullptr,0,2,1);
    const std::int64_t axis=-1;
    const Ort::OpAttr attribute("axis",&axis,1,ORT_OP_ATTR_INT);
    softmax=Ort::Op::Create(info,"Softmax","",13,constraints,types,1,&attribute,1,1,1);
  }
  OrtStatusPtr ComputeV2(OrtKernelContext* raw) noexcept {
    try {
      Ort::KernelContext context(raw);
      const auto q=context.GetInput(0), k=context.GetInput(1), v=context.GetInput(2);
      auto shape=q.GetTensorTypeAndShapeInfo().GetShape();
      const auto ks=k.GetTensorTypeAndShapeInfo().GetShape(), vs=v.GetTensorTypeAndShapeInfo().GetShape();
      if(shape.size()!=4 || ks.size()!=4 || vs!=shape || shape[0]<1 || shape[0]>64 ||
         shape[1]<1 || shape[1]>32 || shape[2]<1 || shape[2]>1024 || shape[3]<1 || shape[3]>128 ||
         ks!=std::vector<std::int64_t>({shape[0],shape[1],shape[3],shape[2]}))
        throw std::invalid_argument("invalid bounded attention Q/K-transpose/V shapes");
      auto output=context.GetOutput(0,shape);
      auto* out=output.GetTensorMutableData<float>();
      const auto* qp=q.GetTensorData<float>();const auto* kp=k.GetTensorData<float>();const auto* vp=v.GetTensorData<float>();
      const auto batches=std::min(patch_batch,shape[0]);
      const std::size_t row_elements=shape[1]*shape[2]*shape[3];
      const std::size_t attention_elements=batches*shape[1]*shape[2]*shape[2];
      auto stream=static_cast<cudaStream_t>(context.GetGPUComputeStream());
      // At most 1 GiB total: 4 patches x 32 heads x 1024^2 x FP32 x two matrices.
      // Frees are ordered on ORT's stream, including on failure after submission.
      Scratch scratch(attention_elements*sizeof(float)*2,stream);
      int device=0;cuda_check(cudaGetDevice(&device));
      Ort::MemoryInfo memory("Cuda",OrtDeviceAllocator,device,OrtMemTypeDefault);
      auto tensor=[&](float* data,std::size_t count,const std::vector<std::int64_t>& dims) {
        return Ort::Value::CreateTensor<float>(memory,data,count,dims.data(),dims.size());
      };
      for(std::int64_t begin=0;begin<shape[0];begin+=patch_batch) {
        const auto count=std::min(patch_batch,shape[0]-begin);
        const std::vector<std::int64_t> qs{count,shape[1],shape[2],shape[3]};
        const std::vector<std::int64_t> kt{count,shape[1],shape[3],shape[2]};
        const std::vector<std::int64_t> logits_shape{count,shape[1],shape[2],shape[2]};
        const auto offset=static_cast<std::size_t>(begin)*row_elements;
        auto q_view=tensor(const_cast<float*>(qp)+offset,count*row_elements,qs);
        auto k_view=tensor(const_cast<float*>(kp)+offset,count*row_elements,kt);
        auto v_view=tensor(const_cast<float*>(vp)+offset,count*row_elements,qs);
        auto out_view=tensor(out+offset,count*row_elements,qs);
        auto logits=tensor(static_cast<float*>(scratch.pointer),attention_elements,logits_shape);
        auto probabilities=tensor(static_cast<float*>(scratch.pointer)+attention_elements,attention_elements,logits_shape);
        const OrtValue* qk[]{q_view,k_view};OrtValue* logits_out[]{logits};
        matmul.Invoke(raw,qk,2,logits_out,1);
        const OrtValue* softmax_in[]{logits};OrtValue* softmax_out[]{probabilities};
        softmax.Invoke(raw,softmax_in,1,softmax_out,1);
        const OrtValue* pv[]{probabilities,v_view};OrtValue* result[]{out_view};
        matmul.Invoke(raw,pv,2,result,1);
      }
      return nullptr;
    } catch(const std::exception& error) { return Ort::GetApi().CreateStatus(ORT_FAIL,error.what()); }
    catch(...) { return Ort::GetApi().CreateStatus(ORT_FAIL,"unknown bounded attention failure"); }
  }
};
struct BoundedAttention : Ort::CustomOpBase<BoundedAttention,Kernel,true> {
  const char* GetName() const {return "BoundedAttention";}
  const char* GetExecutionProviderType() const {return "CUDAExecutionProvider";}
  std::size_t GetInputTypeCount() const {return 3;}
  ONNXTensorElementDataType GetInputType(std::size_t) const {return ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT;}
  std::size_t GetOutputTypeCount() const {return 1;}
  ONNXTensorElementDataType GetOutputType(std::size_t) const {return ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT;}
  OrtStatusPtr CreateKernelV2(const OrtApi& api,const OrtKernelInfo* info,void** output) const noexcept {
    try {*output=new Kernel(info);return nullptr;}
    catch(const std::exception& error) {return api.CreateStatus(ORT_FAIL,error.what());}
    catch(...) {return api.CreateStatus(ORT_FAIL,"unknown bounded attention initialization failure");}
  }
  static Ort::Status InferOutputShape(Ort::ShapeInferContext& context) {
    const auto& shape=context.GetInputShape(0);
    // ORT may call this before symbolic dimensions have been specialized. Do
    // not publish negative integer dimensions (e.g. -1) as concrete shapes.
    for(const auto& dimension:shape)
      if(dimension.IsInt() && dimension.AsInt()<0)return Ort::Status{nullptr};
    return context.SetOutputShape(0,shape);
  }
};
}
extern "C" OrtStatus* ORT_API_CALL RegisterCustomOps(OrtSessionOptions* options,const OrtApiBase* base) {
  const auto* api=base->GetApi(ORT_API_VERSION);
  Ort::InitApi(api);
  try {
    static BoundedAttention op;
    static Ort::CustomOpDomain domain("com.neuromesh");
    static std::once_flag registered;
    std::call_once(registered,[&]{domain.Add(&op);});
    return api->AddCustomOpDomain(options,domain);
  } catch(const std::exception& error) {return api->CreateStatus(ORT_FAIL,error.what());}
  catch(...) {return api->CreateStatus(ORT_FAIL,"bounded attention registration failed");}
}
