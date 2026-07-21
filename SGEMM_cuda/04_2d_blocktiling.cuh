#pragma once
#include <cstdio>
#include <cstdlib>

#include <cublas_v2.h>
#include <cuda_runtime.h>

//the whole motivation of the method in this kernel is to make the ratio of FLOPs to bits fetched from SMEM higher
//meaning we will do more work per thread , we can choose to give a thread a 1d tile or 2d tile , the idea is the same so just implementing the full version is better
//a lot of SMEM load instructions will cause a lot of warps to be in a state of "stall MIO throttle" which is basically warp scheduler  wants to dispatch a new load instruction
//butt the MIO queue is full (it basically hold instructions abotu SMEM loading, dynamic branches, very specific math instruciton (rsqrt,sin,cos,log))
//M*K @ K*N = M*N
template<const uint BM, const uint BK, const uint BN, const uint TM, const uint TN>
__global__ void sgemm_04(int M, 
                         int N, 
                         int K, 
                         float prod_scale, 
                         float sum_scale,
                         const float *A,
                         const float *B,
                         float *C) {

//keep a place where u are going to accumulate the elements
//init SMEM of A and B
//loop over BK
    //populate SMEM
    //loop over TM
        //loop over TN
            //accumulate here
//return result to here
}