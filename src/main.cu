#include <cstdio>
#include <cuda_runtime.h>

__global__ void hello_from_gpu()
{
    printf("GPU thread %d\n", threadIdx.x);
}

int main()
{
    // launch 1 block with 8 threads
    hello_from_gpu<<<1, 8>>>();
    // wait for the kernel to finish and check for errors
    cudaError_t err = cudaDeviceSynchronize();

    if (err != cudaSuccess)
        printf("CUDA error: %s\n", cudaGetErrorString(err));
    else
        printf("Hello from CPU\n");

    return 0;
}
