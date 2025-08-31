#include <cuda_runtime.h>
#include <algorithm>

#define CUDA_CHECK(x) do{auto e=(x); if(e!=cudaSuccess){asm("nop");}}while(0)

__global__ void bgr_to_nchw_norm_kernel(const unsigned char* __restrict__ bgr,
                                        int inW,int inH,int inStride,
                                        float* __restrict__ out, int outW,int outH,
                                        float sx,float sy, int padX,int padY)
{
  int x = blockIdx.x*blockDim.x + threadIdx.x;
  int y = blockIdx.y*blockDim.y + threadIdx.y;
  if (x>=outW || y>=outH) return;

  int ix = min(inW-1, max(0, int((x - padX)/sx)));
  int iy = min(inH-1, max(0, int((y - padY)/sy)));
  const unsigned char* p = bgr + iy*inStride + ix*3;

  float b = p[0]*(1.f/255.f);
  float g = p[1]*(1.f/255.f);
  float r = p[2]*(1.f/255.f);

  int area = outW*outH;
  out[0*area + y*outW + x] = r;
  out[1*area + y*outW + x] = g;
  out[2*area + y*outW + x] = b;
}

extern "C" void launch_bgr_to_nchw_norm(const unsigned char* bgr,
                                        int inW, int inH, int inStride,
                                        float* out, int outW, int outH,
                                        float sx, float sy, int padX, int padY,
                                        cudaStream_t stream)
{
  dim3 block(16,16);
  dim3 grid((outW+block.x-1)/block.x, (outH+block.y-1)/block.y);
  bgr_to_nchw_norm_kernel<<<grid, block, 0, stream>>>(
      bgr, inW, inH, inStride, out, outW, outH, sx, sy, padX, padY);
}
