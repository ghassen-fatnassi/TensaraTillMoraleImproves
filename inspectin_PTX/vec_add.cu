__global__ void vecAdd(float* a, float* b, float* c, int n)
{
    int index = threadIdx.x + blockIdx.x * blockDim.x;
    if (index > n)
    {
        
    }
    else{
        c[index] = a[index] + b[index];
    }
}
 
 
void vecAddLauncher(float* a, float* b, float* c, int n)
{
    // Utility function for launching the kernel
}
 
 
int main()
{
    // Main function of the application
}