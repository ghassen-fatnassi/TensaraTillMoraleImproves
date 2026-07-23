#pragma once
#include <cstdio>
#include <cstdlib>
//this kernel is peak,tiling as an idea is obvious from a first read ,but 2ndand 3rd read tell you how much u didn't know
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

    float reg_tile_result[TM][TN]={0.0};
    float reg_A[TM]={0.0};
    float reg_B[TN]={0.0};

    const uint cRow = blockIdx.y;
    const uint cCol = blockIdx.x;

    //coords of a thread that's responsible one 2D tile of the current block
    const uint threadCol = threadIdx.x / (BN/TN);
    const uint threadRow = threadIdx.x % (BN/TN);

    const uint threads_per_block= (BM*BN)/(TM*TN);//easy to infer
    //look at it 1D it's easy
    //these pointers are pointing to where the work will start to populate the results of the current C block
    A += cRow * BM * K;                     
    B += cCol * BN;                        
    C += cRow * BM * N + cCol * BN;

    //init SMEM of A and B
    __shared__ float shmemA[BM*BK];
    __shared__ float shmemB[BK*BN];//keep row-major

    //you always always have to think from threadIdx.x to the data and to the computation seperately, difference between heaven and hell in mind-mapping
    //these indices are the whole game , pay attention if you are reading my repo
    const uint in_rowA = threadIdx.x / BK;
    const uint in_colA = threadIdx.x % BK;
    const uint offsetA_row_wise = threads_per_block/BK;

    const uint in_rowB= threadIdx.x / BN;
    const uint in_colB= threadIdx.x % BN;
    const uint offsetB_row_wise= threads_per_block/BN;
    //it's very important to watch ur indices because it's what decides coalescing , u need to make surethat lockstep loading from warps are contiguous
    //loop over BK
    for (blk_idx=0; blk_idx<=(K/BK); blk_idx++){
        //even the shmem populating needs to happen in a loop
        //the loop since eachthread is responsible for a 2D tile so each thread is also responsible for loaidng thatpart
        for(uint i=0;i<BM;i+=offsetA_row_wise){
            shmemA[(in_rowA+i)*BK+in_colA]=A[(in_rowA+i)*K+in_colA];// *K is very smart to wrap around
        }
        for(uint i=0;i<BK;i+=offsetB_row_wise){
            shmemB[(in_rowA+)*BN+in_colB]=B[(in_rowB+i)*N+in_colB];// *N is also here for a wraparound
        }
        __syncthreads(); // the usual
        A+=BK;
        B+=BK*N;//we can safely advance them here without affecting next computatin since we only need them when loading to SMEM
        //loop the number of time to do the outer product to accumulate
        // think of it as the operation needing to preserve its lower bound complexity
        // we are doing matmul between 2 small tiles: TM,BK and BK,TN , in our case they are all the same value
        // let's call them just n,n * n,n meaning o(n^3)
        // outer product being o(n²)(the 2 nexted inner loops) then we need another outer loop of accumulation to match the complexity
        for(int i=0;i<BK;++i){
            for(j=0;j<TM;++j){
                reg_A[j]=shmemA[];
            }
            for(j=0;j<TN;++j){
                reg_B[j]=shmemB[];
            }
            //2 loops since this is outer product
            //this is the heart of reducing the number of loads from SMEM ( which caused the MIO throttle)
            for(){
                for(){
                    reg_tile_result[]=reg_A[]*reg_B[];
                }
            }
        }
        __syncthreads();
    }
    for(){
        for(){
            C[]=prod_scale*reg_tile_result[]+sum_scale*C[];
        }
    }
}