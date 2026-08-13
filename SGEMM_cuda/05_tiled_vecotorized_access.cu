#pragma once
#include <cstdio>
#include <cstdlib>
//this kernel is peak,tiling as an idea is obvious from a first read ,but 2ndand 3rd read tell you how much u didn't know
#include <cublas_v2.h>
#include <cuda_runtime.h>

__global__ void kernel()
{
    asm volatile(
        "barrier.cluster.arrive;"
    );
}