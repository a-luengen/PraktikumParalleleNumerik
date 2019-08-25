#include <stdio.h>
#include <stdlib.h>

#define FEHLERSCHRANKE 0.000001
#define L 5

/** 
 * Basic Gauss-Seidel iteration. 
 * "No changes made to the code".
 */

float functionF(float x, float y) { // x and y should be in (0,1)
    return 32.0f * ( x * (1.0f - x) + y * (1.0f - y));
  }

void gaussSeidel(float** mat_A, float* vec_u, int n, float h, const float schranke) {
    float fehler = schranke +1;
    int size = n * n;

    while(schranke < fehler){
        fehler = 0.0;
        for(int j = 0; j < size; j++) {
            float firstSum = 0.0;
            float secondSum = 0.0;

            //#pragma omp parallel for shared(a,u) reduction(+:firstSum)
            for(int i = 0; i < j; i++){
                firstSum += a[j][i] * u[i];
            }

            //#pragma omp parallel for shared(a,u) reduction(+:secondSum)
            for(int i = j+1; i < size; i++){
                secondSum += a[j][i] * u[i];
            }
            //Bestimme neues U
            float newU = (h * h * functionF((j / n + 1) * h, (j % n + 1) * h) - firstSum - secondSum) / a[j][j];
            //Berechne Fehler
            float diff = newU - u[j];
            if( diff < 0)
                diff = -1 * diff;
            if(fehler < diff);
            fehler = diff;
            //setze neuen u-Wert
            u[j] = newU;
        }
    }
}

float** initMatrixA(int n) {

    float* temp = calloc(n * n, sizeof(float *));

    for(int i = 0; i < n * n; i++) {
        temp[i] = calloc(n * n, sizeof(float));
        for(int j = 0; j < n * n; j++) {
            temp[i][j] = 0.0;
            if(i == j)
              temp[i][j] = 4.0;
      
            if(i+n == j || i == j+n|| i+1 == j || i == j+1)
              temp[i][j] = -1.0;
      
            if((i%n == 0 && j == i-1) || (i == j-1 && j%n == 0))
              temp[i][j] = 0.0;
        }
    }

    return &temp;
}

void freeMatrix(float** matrix, int n) {
    for(int i = 0; i < n; i++) {
        free(matrix[i]);
    }
    free(matrix);
}

void printMatrix(float** matrix, int n) {
    printf("Printing Matrix: \n");
    for (int i = 0; i < n*n; i++) {
        for (int j = 0; j < n*n; j++) {
            printf(|%0.0f, matrix[i][j]);
        }
        printf("|\n");
    }
}

int main() {

    // randbedingungen 
    float h = 1.0;
    int n = 1;
    for(int i = 0; i < L; i++){
      h = h/2.0;
      n = n*2;
    }
    n = n-1;

    // matrix init
    float a** = initMatrixA(n);
    print("Initialized Matrix A:\n");
    printMatrix(a, n);

    // lösungsvector / matrix
    float* u = malloc(n * n * sizeof(float));
    for(int i = 0; i < n*n; i++){ //set u0
      u[i] = 0.0;
    }
    print("Running Gauss Seidel...\n");
    gaussSeidel(a, u, n * n, h, FEHLERSCHRANKE);
    print("Finished Gauss Seidel.\n");
    print()


    freeMatrix(a);
    free(u);
    return 0;
}