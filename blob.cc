#include "blob.h"
//#include "tensor.h"

#ifdef WITH_CUDA
#include <caffe2/core/context_gpu.h>
#endif

namespace caffe2 {

TensorCPU BlobUtil::Get() {
#ifdef WITH_CUDA
  if (blob_.IsType<TensorCUDA>()) {
    return TensorCPU(blob_.Get<TensorCUDA>());
  }
#endif
  return blob_.Get<TensorCPU>();
}

void BlobUtil::Set(const TensorCPU &value, bool force_cuda) {
#ifdef WITH_CUDA
  if (force_cuda || blob_.IsType<TensorCUDA>()) {
    //auto tensor = blob_.GetMutable<TensorCUDA>();
	auto tensor =  caffe2::BlobGetMutableTensor(blob_,DeviceType::CUDA);
    tensor->CopyFrom(value);
    return;
  }
#endif
  //auto tensor = blob_.GetMutable<TensorCPU>();
  auto tensor = caffe2::BlobGetMutableTensor(&blob_,DeviceType::CPU);
  tensor->ResizeLike(value);
  tensor->ShareData(value);
}

template<typename T>
void BlobUtil::Set(const std::vector<int> &dim, const std::vector<T> &data, bool force_cuda) {
#ifdef WITH_CUDA
  if (force_cuda || blob_.IsType<TensorCUDA>()) {
    //auto tensor = blob_.GetMutable<TensorCUDA>();
	auto tensor =  caffe2::BlobGetMutableTensor(blob_,DeviceType::CUDA);
    tensor->CopyFrom(value);
    return;
  }
#endif
  auto value = Tensor(dim,DeviceType::CPU);

  	  int count = 0;
  	  for (auto& v : data) {
  		  value.mutable_data<T>()[count] =v;
  		  count++;
  	  }

  auto tensor = caffe2::BlobGetMutableTensor(&blob_,DeviceType::CPU);
  tensor->ResizeLike(value);
  tensor->ShareData(value);
}

void BlobUtil::Print(const std::string &name, int max) {
  auto tensor = Get();
  //I took out 11 07 2018 TensorUtil(tensor).Print(name, max);
}

//or you could just remove the below and place put the function implementation inside the header file and delete it from the cpp file
//see this https://stackoverflow.com/questions/10632251/undefined-reference-to-template-function for better explaination
template void BlobUtil::Set<float>(const std::vector<int> &, const std::vector<float> &, bool);
template void BlobUtil::Set<int>(const std::vector<int> &, const std::vector<int> &, bool);

}  // namespace caffe2
