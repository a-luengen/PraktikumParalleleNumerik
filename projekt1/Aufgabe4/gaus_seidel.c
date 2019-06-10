#include <stdlib.h>
#include <stdio.h>
#include <math.h>

/**
 * gcc: gcc -Wall -g -fopenmp -o gaus_seidel gaus_seidel.c -lm -D PRINT
 * 
 */

#define L 3
#define MAX_K 10

float f(float x, float y);
float u(float x, float y);
int equals(float x, float y);
int a(int x, int y);
int T(int x, int y);


float h = 0.0;
int n = 0;

int main(void) {
    
    h = 1.0 / pow(2.0, L);
    n = ((int) pow(2, L)) - 1;


    
    float* u = malloc(sizeof(float) * n * n);
    // wähle Startvektor u0 aus IR^n 

    for(int i = 0; i < n; i++)
        u[i] = 1.0;

    float sum1 = 0.0;
    float sum2 = 0.0;
    int i;

    for(int k = 0; k < MAX_K; k++) {
        for (int j = 1; j < n; j++) {
            
            // first sum
            for(i = 1; i < j - 1; i++)
                sum1 += a(j,i) * u[i + (k+1) * n];
            
            // second sum
            for(i = j + 1; i < n; i++)
                sum2 += a(j,i) * u[i + k * n];
            

            u[j + (k + 1) * n] = (1.0 / a(j,j) ) * ( pow(h, 2) * f(j,j) - sum1 - sum2);

        }
    }

    #ifdef PRINT
        printf("Vector/Matrix u: \n");
        for(int row = 0; row < n; row++) {
            for(int col = 0; col < n; col++) {
                printf("|%0.8f", u[row + col * n]);
            }
            printf("|\n");
        }
    #endif

    free(u);
    return 0;
}


int equals(float x, float y) {
    #ifdef DEBUG 
        printf("Compare: %0.8f , %0.8f epsilon = %0,8f \n", x, y, h);
    #endif

    if(fabsf(x - y) >= h) {
        return 0;
    }
    return 1;
}


float u(float x, float y) {
    if(equals(x, 0.0) || equals(x,1.0) || equals(y, 0.0) || equals(y, 1.0))
        return 0.0;
    return 16.0 * x * (1.0 - x) * y * (1.0 - y);
}

float f(float x, float y) {
    return 32.0 * (x * (1.0 - x) + y * (1.0 - y));
}

int a(int x, int y) {

    // 1. Check if in T Matrix:
    if((x / n) == (y / n)) {
        // they are in T Matrix
        return T(x % n, y % n);
    }

    // 2. Check if in -I Matrix
    if(abs((x / n) - (y / n)) == 1) {
        // They are in -I Matrix
        if(x % n == y % n){
            return -1;
        }
    }
    // None of the above, return 0
    return 0;
}

int T(int x, int y) {
    if(x == y) return 4;
    if(abs(x - y) == 1) return -1;
    return 0;
}

