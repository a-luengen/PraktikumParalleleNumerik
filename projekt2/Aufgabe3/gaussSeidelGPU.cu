#include <stdio.h>

#include <stdlib.h>

#define FEHLERSCHRANKE 0.000001
//Exponent der Verfeinerung

float functionF(float x, float y);
float** allocateSquareMatrix(int size, int initialize, int n);
float* allocateVector(int size, int initialize);
void printSquareMatrix(float** matrix, int dim);
void printVector(float* vector, int length);
void freeSquareMatrix(float** matrix, int dim);


void gaussSeidel(int n, float fehlerSchranke, float h, float** a, float* u) {
	//TODO: Timeranfang
    float fehler = fehlerSchranke + 1;
    float diff = 0.0;
    float newU = 0.0;
    int n_emb = n + 2;
    float tempSum = 0.0;

    // embedd vector u for corner case
    float** u_emb = allocateSquareMatrix(n_emb * n_emb, 0, n_emb);

    for(int i = 0; i < n_emb; i++) {
        for(int j = 0; j < n_emb; j++) {
            if(j == 0 || i == 0 || j == n_emb || i == n_emb) {
                // fill up with edge value
                u_emb[i][j] = 0.0;
            } else {
                // copy value from u
                u_emb[i][j] = u[i - 1 + n * j - 1];
            }
        }
    }
    // print embedded vector u
    printSquareMatrix(u_emb, n_emb);

    while(fehlerSchranke < fehler){
        fehler = 0.0;

        // using jacoby calculation for gaussSeidel with red-black chess structure
        // iterating over dimensions of u but use n_emb and increment i/j to access values in embedded vector
        // "black" colored elements first
        for(int j = 0; j < n*n; j += 2 ) {
            
            // transform j into coordinates i_emb, j_emb where i_emb = column and j_emb = row in embedded vector/matrix
            // i_emb = j%n + 1
            // j_emb = j/n + 1
            int i_emb, j_emb;
            i_emb = j%n + 1;
            j_emb = j/n + 1;

            // top element
            tempSum = u_emb[i_emb][j_emb - 1];
            // left element
            tempSum += u_emb[i_emb - 1][j_emb];
            // right element
            tempSum += u_emb[i_emb + 1][j_emb];
            // bottom element
            tempSum += u_emb[i_emb][j_emb + 1];

            // calc new value for u
            newU = (h * h * functionF((j / n+1) * h, (j % n+1 )*h) - tempSum) / 4.0;

            // Calculate error
            diff = newU - u_emb[i_emb][j_emb];
            if( diff < 0)
                diff = -1* diff;
            if(fehler < diff);
                fehler = diff;

            //set new value for u in embedded vector
            u_emb[i_emb][j_emb] = newU;
        }

        // "red" colored elements second

        for(int j = 1; j < n*n; j += 2) {
            // transform j into coordinates i_emb, j_emb where i_emb = column and j_emb = row in embedded vector/matrix
            // i_emb = j%n + 1
            // j_emb = j/n + 1
            int i_emb, j_emb;
            i_emb = j%n + 1;
            j_emb = j/n + 1;

            // top element
            tempSum = u_emb[i_emb][j_emb - 1];
            // left element
            tempSum += u_emb[i_emb - 1][j_emb];
            // right element
            tempSum += u_emb[i_emb + 1][j_emb];
            // bottom element
            tempSum += u_emb[i_emb][j_emb + 1];
            // calc new value for u
            newU = (h * h * functionF((j / n+1) * h, (j % n+1 )*h) - tempSum) / 4.0;

            // Calculate error
            diff = newU - u_emb[i_emb][j_emb];
            if( diff < 0)
                diff = -1* diff;
            if(fehler < diff);
                fehler = diff;
            //set new value for u in embedded vector
            u_emb[i_emb][j_emb] = newU;
        }
        /*
        for(int j = 0; j < n*n; j++) {
            float firstSum = 0.0;
            float secondSum = 0.0;
    
            for(int i = 0; i < j; i++){
                firstSum += a[j][i] * u[i];
            }
    
            for(int i = j+1; i < n*n; i++){
                secondSum += a[j][i] * u[i];
            }
            //Bestimme neues U
            newU = (h*h*functionF((j/n+1)*h,(j%n+1)*h) - firstSum - secondSum) / 4.0;
            //Berechne Fehler
            diff = newU-u[j];
            if( diff < 0)
                diff = -1* diff;
            if(fehler< diff);
                fehler = diff;
            //setze neuen u-Wert
            u[j] = newU;
        }
        */
    }
    
    // print embedded vector u
    printSquareMatrix(u_emb, n_emb);

    // get values out of embedded vector
    for(int i = 0; i < n; i++) {
        for(int j = 0; j < n; j++) {
            u[i + j*n] = u_emb[i + 1][j + 1];
        }
    }
    freeSquareMatrix(u_emb, n_emb);
    free(u_emb);
}

int main() {
    //Randbedingungen
    float h = 1.0;
    int n = 1;
    // calc 2^L
    for(int i = 0; i < L; i++){
        n = n * 2;
    }
    h = 1.0 / (float) n;
    n = n - 1;

    printf("h = %f, n = %d, l = %d\n", h, n, L);

    //Lösungsvektoren u
    float *u = allocateVector(n*n, 1);

    //Matrix A
    float **a = allocateSquareMatrix((n*n), 1, n);

    #ifdef PRINT
    printSquareMatrix(a, (n * n));
    printVector(u, (n * n));
    #endif

    // executing gauss seidel verfahren
    gaussSeidel(n, FEHLERSCHRANKE, h, a, u);

    #ifdef PRINT
    printSquareMatrix(a, (n * n));
    printVectorInBlock(u, (n * n), n);
    printVector(u, (n * n));
    #endif

    freeSquareMatrix(a, (n * n));
    free(a);
    free(u);
    return 0;
}

float functionF(float x, float y) { // x and y should be in (0,1)
  return 32.0f*(x*(1.0f-x) + y*(1.0f-y));
}

float** allocateSquareMatrix(int size, int initialize, int n) {
    float** tmp = (float**) malloc(size * sizeof(float*));

    for(int i = 0; i < size; i++) {
        tmp[i] = (float*) malloc(size * sizeof(float));
    }

    if(initialize) {
        for(int i = 0; i < size ;i++){
            for(int j = 0; j < size; j++){
                tmp[i][j] = 0.0;
                if(i == j)
                    tmp[i][j] = 4.0;

                if(i+n == j || i == j + n|| i + 1 == j || i == j+1)
                    tmp[i][j] = -1.0;

                if((i%n == 0 && j == i-1) || (i == j-1 && j%n == 0))
                    tmp[i][j] = 0.0;
            }
        }
    } else {
        for(int i = 0; i < size; i++) {
            for(int j = 0; j < size; j++) {
                tmp[i][j] = 0.0;
            }
        }
    }
    return tmp;
}

/**
 *  Only frees the "rows" of the allocated Matrix. 
 *  Still have to call free on pointer of pointers
 */
void freeSquareMatrix(float** matrix, int dim) {
    for(int i = 0; i < dim; i++) {
        free(matrix[i]);
    }
}

float* allocateVector(int size, int initialize) {
    float* tmp = (float*) malloc(size * sizeof(float));
    if(initialize) {
        for(int i = 0; i < size; i++) {
            tmp[i] = 0.0;
        }
    }
    return tmp;
}
void printSquareMatrix(float** matrix, int dim) {
    printf("Printing sqare matrix with dim = %d\n", dim);
    for(int i = 0; i < dim; i++) {
        for(int j = 0; j < dim; j++) {
            printf("|%f", matrix[i][j]);
        }
        printf("|\n");
    }
}

void printVector(float* vector, int length) {
    printf("Printing Vector with length = %d\n", length);
    for(int i = 0; i < length; i++)
        printf("|%f", vector[i]);
    
    printf("|\n");
}

void printVectorInBlock(float* vector, int length, int blockLength) {
    printf("Printing Vector with length = %d\n", length);
    for(int i = 0; i < length / blockLength; i++) {
        for(int j = 0; j < blockLength; j++) {
            printf("|%f", vector[i + blockLength * j]);
        }
        printf("|\n");
    }
}