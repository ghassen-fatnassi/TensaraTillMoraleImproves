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

//coords of block in C that we will be filling, thinking block wise helps easethe problem    
const uint cRow = blockIdx.x;
const uint cCol = blockIdx.y;

__shared__ float shmemA[BLOCKSIZE * BLOCKSIZE];
__shared__ float shmemB[BLOCKSIZE * BLOCKSIZE];

//we are using 1D blocks , meaning threadIdx has only x , to assure we can see the warps that are next to each other and still assure coalescing
const uint threadx = threadIdx.x / BLOCKSIZE;
const uint thready = threadIdx.x % BLOCKSIZE;

//look at it 1D it's easy
A += cRow * BLOCKSIZE * K;                     
B += cCol * BLOCKSIZE;                        
C += cRow * BLOCKSIZE * N + cCol * BLOCKSIZE;

for(int Blk_idx=0; Blk_idx<K; Blk_idx+=BLOCKSIZE){

    shmemA[threadx*BLOCKSIZE+thready]=A[threadx*K+thready];
    shmemB[threadx*BLOCKSIZE+thready]=B[threadx*N+thready];
    
    __syncthreads(); //syncing to assure no WAR (write after read) happens (thread writing to shmem before the acc op reads from it)
    for(int i=0;i<BLOCKSIZE;++i){
        acc+=shmemA[x*K+y]*shmemB[X*N+y];
    }
    __syncthreads();
    A+=BLOCKSIZE;//this is necessary, it moves the A pointer to cover the block and eveything down/right
    B+=BLOCKSIZE*N;
    // the latest sync threads is to avoid data race,i.e some threads overwriting the shmem while some threads are still doing acc
    //basically loop in lockstep (since any extra iteration from a thread will overwrite a shmem that another thread might be using when doing the inner loop)
}
C[threadx*N+thready]=prod_scale*acc+sum_scale*C[threadx*N+thready];

}