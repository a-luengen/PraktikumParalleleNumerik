#include <stdio.h>
#include <math.h> 
#include <omp.h>
#include <stdlib.h>
#include <time.h>

#define N 10

typedef struct Matrix {
    int rows, columns;
    float ** matrix;
} Matrix;

void matMul(Matrix, Matrix, Matrix);
void matMulIKJ(Matrix, Matrix, Matrix);
void matMulIJK(Matrix, Matrix, Matrix);
void matMulOmp(Matrix, Matrix, Matrix);
Matrix createRandomMatrix(int rows, int cols);
void freeMatrix(Matrix m);
void printMatrix(Matrix m);
void clearMatrix(Matrix m);

int main(void) {
    // seed random generator
    srand((unsigned int)time(NULL));
    
    Matrix a = createRandomMatrix(N, N);
    Matrix b = createRandomMatrix(N, N);
    Matrix c = createRandomMatrix(N, N);

#ifdef PRINT
    printf("Done \n");
    printf("Printing matrix a:\n");
    printMatrix(a);
    printf("Print Matrix b:\n");
    printMatrix(b);
    printf("Print Matrix c:\n");
    printMatrix(c);

    printf("Calculating Matrix Multiplication  a * b = c \n");
#endif
    matMul(a, b , c);
#ifdef PRINT
    printf("Print result in Matrix c:\n");
    printMatrix(c);
    printf("Done!\n");
#endif
    freeMatrix(a);
    freeMatrix(b);
    freeMatrix(c);
    return 0;
}

void matMul(Matrix a, Matrix b, Matrix c) {
    //matMulIJK(a,b,c);
    //matMulIKJ(a, b, c);
    matMulOmp(a, b, c);
}

void matMulOmp(Matrix a, Matrix b, Matrix c) {
    printf("Doing OMP with IKJ and temp-var: \n");
    
    int i, j, k;
    float temp = 0.0;

    #pragma omp parallel for private(i, j, k, temp) schedule(dynamic, 1)
    for(i = 0; i < a.rows; i++) {
        for(k = 0; k < a.columns; k++) {
            temp = a.matrix[i][k];
            for (j = 0; j < b.columns; j++) {
                 c.matrix[i][j] += temp * b.matrix[k][j];
            }
        }
    }
}

void matMulIJK(Matrix a, Matrix b, Matrix c) {
    printf("Doing ijk-Algorithmus: \n");
    int i, j, k;
    #pragma omp parallel for private(i, j, k)
    for(i = 0; i < a.rows; i++) {
        for (j = 0; j < b.columns; j++) {
            for(k = 0; k < a.columns; k++) {
                c.matrix[i][j] += a.matrix[i][k] * b.matrix[k][j];
            }
        }
    }
}

void matMulIKJ(Matrix a, Matrix b, Matrix c) {
    printf("Doing ikj-Algorithmus: \n");
    int i, j, k;
    for(i = 0; i < a.rows; i++) {
        for(k = 0; k < a.columns; k++) {
            for (j = 0; j < b.columns; j++) {
                c.matrix[i][j] += a.matrix[i][k] * b.matrix[k][j];
            }
        }
    }
}

Matrix createRandomMatrix(int rows, int cols){
    
    // allocate matrix with all rows
    Matrix temp = {rows, cols, calloc(rows, sizeof(float *))};

    if (temp.matrix == NULL) {
        printf("panic\n");
    }

    // allocate collumn for each row
    for (int i = 0; i < rows; i++) {
        temp.matrix[i] = calloc(cols, sizeof temp.matrix[i][0]);

        for(int j = 0; j < cols; j++) {
            temp.matrix[i][j] = ((float)rand()/(float)(RAND_MAX)) * 5.0;
        }
    }

    return temp;
}

void freeMatrix(Matrix m) {
    // free column for each row
    for(int i = 0; i < m.rows; i++) {

        free( m.matrix[i] );
    }
    free( m.matrix );
}

void printMatrix(Matrix m) {
    for (int i = 0; i < m.rows; i++)
    {
        for (int j = 0; j < m.columns; j++)
        {
            printf("|%0.5f", m.matrix[i][j]);
        }
        printf("|\n");
    }
}

void clearMatrix(Matrix m) {
    for (int i = 0; i < m.rows; i++) {
        for (int j = 0; j < m.columns; j++) {
            m.matrix[i][j] = 0.0;
        }
    }
}