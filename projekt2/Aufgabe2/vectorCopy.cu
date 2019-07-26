#include <time.h>
#include <stdio.h>

#define N 10000000

__global__ void increment(int *array, int length, int x) {
    int idx = blockIdx.x*blockDim.x + threadIdx.x;
    if (idx < length) {
        array[idx] = array[idx] + x;
    }
}

int main(int x) {
    int bytes = N*sizeof(int);
    int* a = (int*)malloc(bytes);
    int* b = (int*)malloc(bytes);

    int* cuda_a;
    int* cuda_b;
    cudaMalloc((int**) &cuda_a, bytes);
    cudaMalloc((int**) &cuda_b, bytes);

    memset(a, 0, bytes);

    clock_t start, end;
    double cpu_time_used;

    // copy the values in array a to array b on the host memory
    start = clock();
    memcpy(b, a, bytes);
    end = clock();

    cpu_time_used = ((double) (end - start)) / (CLOCKS_PER_SEC/1000);
    printf("%f\n", cpu_time_used);

    cudaMemcpy(cuda_a, b, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(cuda_b, cuda_a, bytes, cudaMemcpyDeviceToDevice);

    // try 4, 8, 16, 24 and 32
    int blocksize = 16;

    dim3 dimBlock(blocksize);
    dim3 dimGrid( ceil(N/(float)blocksize) );
    increment <<<dimGrid,dimBlock>>> (cuda_b, N, x);

    cudaMemcpy(a, cuda_b, bytes, cudaMemcpyDeviceToHost);

    cudaFree(cuda_a);
    cudaFree(cuda_b);
    free(a);
    free(b);
}