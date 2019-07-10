#include <stdio.h>
#include <omp.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#define N 10000000

int main() {
    int bytes = N*sizeof(int);
    int* a = (int*)malloc(bytes);

    memset(a, 0, bytes);

    clock_t start, end;
    double cpu_time_used;

    start = clock();
    #pragma omp parallel for 
    for(int i=0; i<N; i++) {
        a[i] = a[i] + 1;
    }
    end = clock();

    cpu_time_used = ((double) (end - start)) / (CLOCKS_PER_SEC/1000);
    printf("%f ms\n", cpu_time_used);

    free(a);

    return 0;
}