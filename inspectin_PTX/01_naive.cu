#pragma once
#include <cstdio>
#include <cstdlib>

#include <cublas_v2.h>
#include <cuda_runtime.h>

//M*K @ K*N = M*N
__global__ void sgemm_01(int M, 
                            int N, 
                            int K, 
                            float prod_scale, 
                            float sum_scale,
                            const float *A,
                            const float *B,
                            float *C) {

const uint x = blockIdx.x * blockDim.x + threadIdx.x;
const uint y = blockIdx.y * blockDim.y + threadIdx.y;


if(x<M && y<N){
    float acc = 0.0f;
    for(int i=0;i<K;++i){
        acc+=A[x*K+i]*B[i*N+y];
    }
    C[x*N+y]=prod_scale*acc+sum_scale*C[x*N+y];
}
}
//no coalescing, loss of warped access(loads)
//no use of cache for accessing B
//no use of shared memory