#include <cuda_runtime.h>
#include "shobo_detectors/kernels.hpp"

namespace {

__global__ void bgr_to_nchw_norm_kernel(const unsigned char* __restrict__ bgr,
                                        int inW, int inH, int inStride,
                                        float* __restrict__ out,
                                        int outW, int outH,
                                        float sx, float sy,
                                        int padX, int padY) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;
  if (x >= outW || y >= outH) return;

  // Minimal no-op: write zeros so we always have defined output
  // (replace with your real conversion)
  int oidx = (0 * outH + y) * outW + x;
  out[oidx] = 0.0f;
  out[oidx + outH * outW] = 0.0f;
  out[oidx + 2 * outH * outW] = 0.0f;
}

} // anonymous

namespace shobo {

void bgr_to_nchw_norm(const unsigned char* bgr,
                      int inW, int inH, int inStride,
                      float* out, int outW, int outH,
                      float sx, float sy, int padX, int padY,
                      cudaStream_t stream) {
  dim3 block(16, 16);
  dim3 grid((outW + block.x - 1) / block.x, (outH + block.y - 1) / block.y);
  bgr_to_nchw_norm_kernel<<<grid, block, 0, stream>>>(
      bgr, inW, inH, inStride, out, outW, outH, sx, sy, padX, padY);
}

} // namespace shobo
