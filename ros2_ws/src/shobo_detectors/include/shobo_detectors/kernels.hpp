#pragma once
#include <cuda_runtime.h>

namespace shobo {

// Host-callable wrapper implemented in kernels.cu
void bgr_to_nchw_norm(const unsigned char* bgr,
                      int inW, int inH, int inStride,
                      float* out,
                      int outW, int outH,
                      float sx, float sy,
                      int padX, int padY,
                      cudaStream_t stream = 0);

} // namespace shobo
