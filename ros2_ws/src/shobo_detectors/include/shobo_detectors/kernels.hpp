#pragma once
#include <cuda_runtime.h>

// Launch the BGR8 -> NCHW [0..1] letterbox normalization on CUDA
void launch_bgr_to_nchw_norm(const unsigned char* bgr,
                             int inW, int inH, int inStride,
                             float* out, int outW, int outH,
                             float sx, float sy, int padX, int padY,
                             cudaStream_t stream);
