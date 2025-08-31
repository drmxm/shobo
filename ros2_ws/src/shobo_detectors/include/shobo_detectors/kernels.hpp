#pragma once
#include <cuda_runtime.h>

extern "C" void launch_bgr_to_nchw_norm(
    const unsigned char* bgr,
    int inW, int inH, int inStride,
    float* out,
    int outW, int outH,
    float sx, float sy,
    int padX, int padY,
    cudaStream_t stream);
