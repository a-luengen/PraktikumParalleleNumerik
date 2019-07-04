#include <stdio.h>

#define ARRAY_SIZE 2000

void printArray(int *array, int length) {
        return;
        for(int i = 0; i < length; i++) {
                printf("|%d", array[i]);
        }
        printf("|\n");
}

int main(void) {

        int dataBits = ARRAY_SIZE * sizeof(int) * 8;

        printf("Copy Between GPU and Host System Test. \n");
        printf("Array Size : %d Entries\n", ARRAY_SIZE);
        printf("Memory Size: %d bytes\n", dataBits / 8);


        cudaEvent_t start, stop;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);
        float ms = 0.0f;


        int *host1 = (int*) malloc(sizeof(int) * ARRAY_SIZE);
        int *host2 = (int*) malloc(sizeof(int) * ARRAY_SIZE);

        //int *gpu1 = (int*) cudaMalloc(sizeof(int) * ARRAY_SIZE);
        //int *gpu2 = (int*) cudaMalloc(sizeof(int) * ARRAY_SIZE);
        int *gpu1, *gpu2;
        cudaMalloc((void**) &gpu1, sizeof(int) * ARRAY_SIZE);
        cudaMalloc((void**) &gpu2, sizeof(int) * ARRAY_SIZE);

        // init host1 array
        for(int i = 0; i < ARRAY_SIZE; i++) {
                host1[i] = i + 1;
        }

        printArray(host1, ARRAY_SIZE);

        // copy on host
        cudaEventRecord(start);

        memcpy(host2, host1, sizeof(int) * ARRAY_SIZE);

        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&ms, start, stop);

        //printArray(host2, ARRAY_SIZE);
        printf("Host1 to Host2 took: %.8fms - %013.2f Bit/s \n", ms, ( dataBits / ms ));


        // copy from host to gpu
        cudaEventRecord(start);

        cudaMemcpy(gpu1, host2, sizeof(int) * ARRAY_SIZE, cudaMemcpyHostToDevice);

        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&ms, start, stop);
        printf("Host2 to GPU1  took: %.8fms - %013.2f Bit/s \n", ms, ( dataBits / ms));

        // copy on gpu
        cudaEventRecord(start);

        cudaMemcpy(gpu2, gpu1, sizeof(int) * ARRAY_SIZE, cudaMemcpyDeviceToDevice);

        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&ms, start, stop);
        printf("GPU1  to GPU2  took: %.8fms - %013.2f Bit/s \n", ms, (dataBits / ms));


        // copy from gpu to host
        cudaEventRecord(start);

        cudaMemcpy(host1, gpu2, sizeof(int) * ARRAY_SIZE, cudaMemcpyDeviceToHost);

        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&ms, start, stop);
        printf("GPU2  to Host1 took: %.8fms - %013.2f Bit/s \n", ms, (dataBits / ms));

        printArray(host1, ARRAY_SIZE);
        return 0;
}
