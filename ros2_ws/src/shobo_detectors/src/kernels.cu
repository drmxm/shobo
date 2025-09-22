#include <cuda_runtime.h>
#include "shobo_detectors/kernels.hpp"

namespace {

__device__ inline float sample(const unsigned char* data, int stride, int x, int y, int c) {
  const unsigned char* row = data + static_cast<size_t>(y) * stride;
  const unsigned char* px = row + static_cast<size_t>(x) * 3;
  return static_cast<float>(px[c]) * (1.f / 255.f);
}

__global__ void bgr_to_nchw_norm_kernel(const unsigned char* __restrict__ bgr,
                                        int inW, int inH, int inStride,
                                        float* __restrict__ out,
                                        int outW, int outH,
                                        float sx, float sy,
                                        int padX, int padY) {
  const int x = blockIdx.x * blockDim.x + threadIdx.x;
  const int y = blockIdx.y * blockDim.y + threadIdx.y;
  if (x >= outW || y >= outH) {
    return;
  }

  const int channelSize = outW * outH;
  const int outIdx = y * outW + x;

  const float fill = 114.f / 255.f;

  const float srcX = (static_cast<float>(x) - static_cast<float>(padX) + 0.5f) / sx - 0.5f;
  const float srcY = (static_cast<float>(y) - static_cast<float>(padY) + 0.5f) / sy - 0.5f;

  if (srcX < 0.f || srcY < 0.f || srcX > static_cast<float>(inW - 1) || srcY > static_cast<float>(inH - 1)) {
    out[outIdx] = fill;
    out[outIdx + channelSize] = fill;
    out[outIdx + 2 * channelSize] = fill;
    return;
  }

  const int x0 = max(min(static_cast<int>(floorf(srcX)), inW - 1), 0);
  const int y0 = max(min(static_cast<int>(floorf(srcY)), inH - 1), 0);
  const int x1 = min(x0 + 1, inW - 1);
  const int y1 = min(y0 + 1, inH - 1);

  const float dx = srcX - static_cast<float>(x0);
  const float dy = srcY - static_cast<float>(y0);

  const float w00 = (1.f - dx) * (1.f - dy);
  const float w10 = dx * (1.f - dy);
  const float w01 = (1.f - dx) * dy;
  const float w11 = dx * dy;

  for (int c = 0; c < 3; ++c) {
    const float v00 = sample(bgr, inStride, x0, y0, c);
    const float v10 = sample(bgr, inStride, x1, y0, c);
    const float v01 = sample(bgr, inStride, x0, y1, c);
    const float v11 = sample(bgr, inStride, x1, y1, c);

    const float value = v00 * w00 + v10 * w10 + v01 * w01 + v11 * w11;
    out[outIdx + c * channelSize] = value;
  }
}

} // namespace

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
