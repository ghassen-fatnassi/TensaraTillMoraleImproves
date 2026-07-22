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
//TM and TN need to be equal or the whole design of ths kernel falls apart , i'm following it for now so i am coherent with siboehm guide , 
//even tho themore i read closely the worse the guide becomes , has valuable ideas tho
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
// we'll keep a tile_result (which is basically per thread result, thread takescare of a 2D tile here)
// so the tile_result=tilesize=TM*TN
float reg_tile_result[TM][TN]={0.0};
//and additionally we have to keep registers to put SMEM data we'll reuuse 
float reg_A[TM]={0.0};
float reg_B[TN]={0.0};


const uint cRow = blockIdx.x;
const uint cCol = blockIdx.y;

//coords of a thread that's responsible one 2D tile of the current block
const uint threadx = threadIdx.x / (BN/TN);
const uint thready = threadIdx.x % (BN/TN);

//look at it 1D it's easy
//these pointers are pointing to where the work will start to populate the results of the current C block
A += cRow * BM * K;                     
B += cCol * BN;                        
C += cRow * BM * N + cCol * BN;

//init SMEM of A and B
__shared__ float shmemA[BM*BK];
__shared__ float shmemB[BK*BN];//keep row-major

//loop over BK
for (blk_idx=0;blk_idx<=(K/BK);blk_idx++){
    //even the shmem populating needs to happen in a loop
    //theloop since eachthread is responsible fora 2D tile so each thread is also responsible for loaidng thatpart
    for(int i=0;i<=TM)
    shmemA[]=A[];
    shmemB[]=B[];
    __syncthreads(); // the usual
    //loop over TM
        //loop over TN
            //accumulate here
    __syncthreads();
}
//return result to here
}