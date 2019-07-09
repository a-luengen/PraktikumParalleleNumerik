#include <stdio.h>

#define ARRAY_SIZE 100000

__global__
void constInc(int increment, int *array, int arrayLength) {
        int i = blockIdx.x * blockDim.x + threadIdx.x;
        if(i < arrayLength) array[i] = array[i] + increment;
}

/**
*  Print an Array with certain length on console
*/
void printArray(int *array, int length) {
        for(int i = 0; i < length; i++) {
                printf("|%d", array[i]);
        }
        printf("|\n");
}

/**
* Increment each element of the given Array with a certain constant on the GPU
*/
void incrementOnGPU(int *array, int arLength, int constant) {
	int blockSize = 32;
	int blocks = 32;
	
	int* gpu_arr;
	cudaMalloc((void***) &gpu_arr, arLength * sizeof(int));

        cudaMemcpy(gpu_arr, array, sizeof(int) * arLength, cudaMemcpyHostToDevice);

        // execute kernel
        constInc <<< blocks, blockSize >>> (constant, gpu_arr, arLength);

        cudaMemcpy(array, gpu_arr, sizeof(int) * arLength, cudaMemcpyDeviceToHost);
}

int main(void) {

        cudaSetDevice(0);

        int dataBits = ARRAY_SIZE * sizeof(int) * 8;

        printf("Copy Between GPU and Host System Test. \n");
        printf("Array Size : %d Entries\n", ARRAY_SIZE);
        printf("Memory Size: %d bytes\n", dataBits / 8);

        int *host1 = (int*) malloc(sizeof(int) * ARRAY_SIZE);

        // init host1 array
        for(int i = 0; i < ARRAY_SIZE; i++) {
                host1[i] = i + 1;
        }

        printArray(host1, ARRAY_SIZE);
        incrementOnGPU(host1, ARRAY_SIZE, 5);
        printArray(host1, ARRAY_SIZE);
        return 0;
}
