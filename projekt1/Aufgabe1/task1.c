/*
 * Compilation:
 * 		w/ image output: gcc -Wall -o task1 -D IMAGE_OUTPUT task1.c -lm
 * 		w/o image output: gcc -Wall -o task1 task1.c -lm
 */

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <omp.h>


int main() {
    
    #pragma omp parallel
    {
        int id = omp_get_thread_num();
        int total = omp_get_num_threads();
        printf("Hi, I'm %d of %d Threads.\n", id, total);
    }
    return 0;
}