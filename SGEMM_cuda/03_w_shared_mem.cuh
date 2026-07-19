#pragma once
#include <cstdio>
#include <cstdlib>

#include <cublas_v2.h>
#include <cuda_runtime.h>

//M*K @ K*N = M*N
template<const uint BLOCKSIZE>
__global__ void sgemm_03(int M, 
                         int N, 
                         int K, 
                         float prod_scale, 
                         float sum_scale,
                         const float *A,
                         const float *B,
                         float *C) {

const uint cRow = blockIdx.x;
const uint cCol = blockIdx.y;

__shared__ float shmemA[BLOCKSIZE * BLOCKSIZE];
__shared__ float shmemB[BLOCKSIZE * BLOCKSIZE];
const uint threadx = threadIdx.x / BLOCKSIZE;
const uint thready = threadIdx.x % BLOCKSIZE;

A += cRow * BLOCKSIZE * K;                    
B += cCol * BLOCKSIZE;                        
C += cRow * BLOCKSIZE * N + cCol * BLOCKSIZE;

for(int Blk_idx=0; Blk_idx<K; Blk_idx+=BLOCKSIZE){

    shmemA[cRow*BLOCKSIZE+cCol]=A[threadx*BLOCKSIZE+thready];
    shmemB[cRow*BLOCKSIZE+cCol]=B[threadx*BLOCKSIZE+thready];
    
    __syncthreads();
    for(int i=0;i<BLOCKSIZE;++i){
        acc+=shmemA[x*K+y]*shmemB[X*N+y];
    }
    __syncthreads();
    // the latest sync threads isn't to avoid data race, it's to avoid some threads overwriting the cache while some threads are still doing acc
}
C[someindex]=prod_scale*acc+sum_scale*C[someindex];

}