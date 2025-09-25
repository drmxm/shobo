// file: ros2_ws/src/shobo_detectors/include/shb/trt_dla_utils.hpp
#pragma once
#ifdef SHB_USE_TRT
#include <NvInfer.h>
#include <iostream>
namespace shb {
inline void enable_dla(nvinfer1::IBuilder* b, nvinfer1::IBuilderConfig* c,
                       int dlaCore=0, bool int8=true, bool fp16=true) {
#ifdef SHB_USE_DLA
  if (!b || !c) return;
  int n = b->getNbDLACores();
  if (n <= 0) { std::cerr << "[TRT] No DLA cores; using GPU.\n"; return; }
  if (dlaCore < 0 || dlaCore >= n) dlaCore = 0;
  using nvinfer1::BuilderFlag; using nvinfer1::DeviceType;
  if (int8) c->setFlag(BuilderFlag::kINT8);
  if (fp16) c->setFlag(BuilderFlag::kFP16);
  c->setDefaultDeviceType(DeviceType::kDLA);
  c->setDLACore(dlaCore);
  c->setFlag(BuilderFlag::kGPU_FALLBACK);
  std::cout << "[TRT] DLA core " << dlaCore
            << " INT8="<<int8<<" FP16="<<fp16<<" + GPU fallback\n";
}
#endif
}
#endif
