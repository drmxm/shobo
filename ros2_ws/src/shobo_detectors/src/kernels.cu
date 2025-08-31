#include <cuda_runtime.h>

__global__ void bgr_to_nchw_norm_kernel(
    const unsigned char* __restrict__ bgr,
    int inW, int inH, int inStride,
    float* __restrict__ out,
    int outW, int outH,
    float sx, float sy,
    int padX, int padY)
{
  const int x = blockIdx.x * blockDim.x + threadIdx.x;
  const int y = blockIdx.y * blockDim.y + threadIdx.y;
  if (x >= outW || y >= outH) return;

  const float srcXf = (x - padX) / sx;
  const float srcYf = (y - padY) / sy;
  const int   srcX  = static_cast<int>(roundf(srcXf));
  const int   srcY  = static_cast<int>(roundf(srcYf));

  float r = 0.f, g = 0.f, b = 0.f;
  if (srcX >= 0 && srcX < inW && srcY >= 0 && srcY < inH) {
    const unsigned char* p = bgr + srcY * inStride + srcX * 3;
    const float inv255 = 1.0f / 255.0f;
    b = static_cast<float>(p[0]) * inv255;
    g = static_cast<float>(p[1]) * inv255;
    r = static_cast<float>(p[2]) * inv255;
  }

  const int o = y * outW + x;
  out[o]                   = r;
  out[outW * outH + o]     = g;
  out[2 * outW * outH + o] = b;
}

extern "C" void launch_bgr_to_nchw_norm(
    const unsigned char* bgr,
    int inW, int inH, int inStride,
    float* out,
    int outW, int outH,
    float sx, float sy,
    int padX, int padY,
    cudaStream_t stream)
{
  dim3 block(16, 16);
  dim3 grid((outW + block.x - 1) / block.x, (outH + block.y - 1) / block.y);
  bgr_to_nchw_norm_kernel<<<grid, block, 0, stream>>>(
      bgr, inW, inH, inStride, out, outW, outH, sx, sy, padX, padY);
}
