#pragma once
#include <cstdio>
#include <cstdlib>

#include <cublas_v2.h>
#include <cuda_runtime.h>

//M*K @ K*N = M*N
template <const uint BLOCKSIZE>
__global__ void sgemm_02(int M, 
                            int N, 
                            int K, 
                            float prod_scale, 
                            float sum_scale,
                            const float *A,
                            const float *B,
                            float *C) {
// in this implementation we'll use a 1D BLock
// in the naive we used dim3 blockDim(32,32,1)
// here we'll call this kernel as a thread from dim3 blockDim(32*32,1,1)
// meaning we'll have threadIdx.y=0 and threadIdx.z=0 for all launched threads
// the reason is to easily map between warps and data they are accessing so we can assure coalescing
// atleast that's my current mental model
// very neat , looking at the block as 1D array is more helpful if you think about coalescing
// templating makes the kernel even more extensible, (extendible? , idk)
const uint  x = blockIdx.x * BLOCKSIZE + threadIdx.x / BLOCKSIZE;
const uint  y = blockIdx.y * BLOCKSIZE + threadIdx.x % BLOCKSIZE;
if(x<M && y<N){
    float acc = 0.0f;
    for(int i=0;i<K;++i){
        acc+=A[x*k+i]*B[i*N+y];
    }
    C[x*N+y]=prod_scale*acc+sum_scale*C[x*N+y];
}
}