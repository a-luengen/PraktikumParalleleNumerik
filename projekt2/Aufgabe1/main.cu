
#include <stdio.h>



int main(void) {
        // print out important data about the gpu
        int nDevices = 0;
        cudaGetDeviceCount(&nDevices);
        printf("Number of Devices: %d\n", nDevices);

        cudaDeviceProp prop;
        int i;
        for(i = 0; i < nDevices; i++) {
                cudaGetDeviceProperties(&prop, i);
                printf("Device Number:                %d\n", i);
                printf("Device Name:                  %s\n", prop.name);
                printf("Memory Clock Rate (KHz):      %d\n", prop.memoryClockRate);
                printf("Memory Bus Width (bits):      %d\n", prop.memoryBusWidth);
                printf("Global Memory size (bytes):   %zu\n", prop.totalGlobalMem);
        }

        return 0;
}
