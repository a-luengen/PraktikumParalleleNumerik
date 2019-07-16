#include <stdio.h>

#include <stdlib.h>

#define FEHLERSCHRANKE 0.000001
//Exponent der Verfeinerung

float functionF(float x, float y);
float* allocateSquareMatrix(int size, int initialize, int n);
float* allocateVector(int size, int initialize);
void printSquareMatrix(float **matrix, int dim);
void printSquareMatrix(float *matrix, int dim);
void printVector(float *vector, int length);
void freeSquareMatrix(float **matrix, int dim);
float calculateError(float *old_val, float *new_val, int dim);
void printVectorInBlock(float *vector, int length, int blockLength);
void checkForError(const char* msg);

const int ITERATE_ON_BLACK = 0;
const int ITERATE_ON_RED = 1;
const int THREADS_PER_BLOCK = 32;
const int BLOCK_DIMENSION = 8;

/**
*   expecting embedded result-matrix to iterate on
*   dim: dimension of the matrix u (without embedding)
*   ITERATION_FLAG: has value 0 to iterate on "black" elements or 1 to iterate on "red" elements
*/
__global__ void redBlackIteration(int dim_u, int dim_u_emb, float h, float* u_emb, int ITERATION_FLAG) {


    int threadID = threadIdx.x * 2 + ITERATION_FLAG;
    int i_offset = blockDim.x * blockIdx.x;
    int j_offset = blockDim.y * blockIdx.y;
    // use index of thread to calculate position in matrix u_emb 
    // to execute computation on
    int j_inner, i_emb, j_emb;

    // 1. Calculate the index's of embedded matrix to a thread has to work on
    j_emb = j_offset + (int) threadID / blockDim.x;
    i_emb = i_offset + (int) threadID % 8 + (1 - 2 * ITERATION_FLAG) * (j_emb % 2);
    
    printf("threadID = %d, j_emb = %d, i_emb = %d\n", threadID, j_emb, i_emb);
    if(i_emb < dim_u_emb - 1 && i_emb > 0 && j_emb < dim_u_emb - 1 && j_emb > 0) {
        // 2. calculate the index's of inner matrix for the functionF-call
        //i_inner = i_emb - 1;
        j_inner = j_emb - 1;
        printf("I calculate - threadID = %d, i_offset = %d, j_offset = %d, j_emb = %d, i_emb = %d, j_inner = %d\n", threadID, i_offset, j_offset, j_emb, i_emb, j_inner);
        // 3. calculate new value for u_emb
        float tempSum = 
            // top element
            u_emb[i_emb + (j_emb - 1) * dim_u_emb] 
            // left element
            + u_emb[i_emb - 1 + j_emb * dim_u_emb] 
            // right element
            + u_emb[i_emb + 1 + j_emb * dim_u_emb] 
            // bottom element
            + u_emb[i_emb + (j_emb + 1) * dim_u_emb]; 

        // calc new value for u
        float newU = (h * h * functionF((j_inner / dim_u + 1) * h, (j_inner % dim_u + 1) * h) + tempSum) / 4.0;
        // 4. replace old value
        u_emb[i_emb + j_emb * dim_u_emb] = newU;
    }
}

/**
*      
*   n: Dimension of u matrix
*   fehlerSchranke: Abbruchbedingung für Gaus Seidel Verfahren
*   h: Verfeinerung/Gitterschrittweite
*   a: pointer to 2D-Array of Matrix a (currently not used, due 
*       to extraction of calculation pattern into algorithm)
*   u: pointer to u matrix
*/
void gaussSeidel(int n, float fehlerSchranke, float h, float *u)
{
    //TODO: Timeranfang
    float fehler = fehlerSchranke + 1;

    // embedd vector u for corner case
    // u(0, y) = u(1, y) = u(x, 0) = u(y, 1) = 0.0
    int n_emb = n + 2;
    float *u_emb = allocateSquareMatrix(n_emb * n_emb, 0, n_emb);
    float *u_emb_new = allocateSquareMatrix(n_emb * n_emb, 0, n_emb);

    for (int i = 0; i < n_emb; i++)
    {
        for (int j = 0; j < n_emb; j++)
        {
            if (j == 0 || i == 0 || j == n_emb || i == n_emb)
            {
                // fill up with edge value
                u_emb[i + j * n_emb] = 0.0;
            }
            else
            {
                // copy value from u
                u_emb[i + j * n_emb] = u[i - 1 + n * j - 1];
            }
        }
    }
#ifdef PRINT
    // print embedded vector u
    printSquareMatrix(u_emb, n_emb);
#endif

    // allocate device memory
    float *gpu_u_emb;
    cudaMalloc((void**)&gpu_u_emb, n_emb * n_emb * sizeof(float));
    // copy from local to device
    cudaMemcpy(gpu_u_emb, u_emb, n_emb * n_emb * sizeof(float), cudaMemcpyHostToDevice);
    checkForError("After Copying data to device.");
    
    // calculate the blocks per dimension
    int blocksPerDimension = 1 + n_emb / BLOCK_DIMENSION;
    dim3 numBlocks(blocksPerDimension, blocksPerDimension);

    printf("Running with numBlocks: %d, %d\n and %d of Threads per Block.\n", blocksPerDimension, blocksPerDimension, THREADS_PER_BLOCK);
    // Iterate as long as we do not come below our fehelrSchranke
    while (fehlerSchranke < fehler)
    {
        //int dim_u, int dim_u_emb, float h, float* u_emb, int ITERATION_FLAG
        // black iteration
        redBlackIteration<<<numBlocks, THREADS_PER_BLOCK>>>(n, n_emb, h, u_emb, ITERATE_ON_BLACK);
        // red iteration
        redBlackIteration<<<numBlocks, THREADS_PER_BLOCK>>>(n, n_emb, h, u_emb, ITERATE_ON_RED);
        cudaDeviceSynchronize();
        checkForError("After 2 Kernel Executions");
        // move result of first iteration onto host
        cudaMemcpy(u_emb_new, gpu_u_emb, n_emb * n_emb * sizeof(float), cudaMemcpyDeviceToHost);
        
        // calculate error
        fehler = calculateError(u_emb, u_emb_new, n_emb);
        // switch pointers
        float temp = *u_emb;
        *u_emb = *u_emb_new;
        *u_emb_new = temp;

        #ifdef PRINT
        printf("Iteration-Error = %0.0f\n", fehler);
        printSquareMatrix(u_emb, n_emb);
        #endif
    }

#ifdef PRINT
    // print embedded vector u
    printSquareMatrix(u_emb, n_emb);
#endif

    // get values out of embedded vector
    for (int i = 0; i < n; i++)
    {
        for (int j = 0; j < n; j++)
        {
            u[i + j * n] = u_emb[i + 1 + (j + 1) * n_emb];
        }
    }
    //freeSquareMatrix(u_emb, n_emb);
    free(u_emb);
}

int main()
{
    //Randbedingungen
    float h = 1.0;
    int n = 1;
    // calc 2^L
    for (int i = 0; i < L; i++)
    {
        n = n * 2;
    }
    h = 1.0 / (float)n;
    n = n - 1;

    printf("h = %f, n = %d, l = %d\n", h, n, L);

    //Lösungsvektoren u
    float *u = allocateVector(n * n, 1);

#ifdef PRINT
    //printSquareMatrix(a, (n * n)*(n * n));
    printVector(u, (n * n));
#endif

    // executing gauss seidel verfahren
    gaussSeidel(n, FEHLERSCHRANKE, h, u);

    printVectorInBlock(u, (n * n), n);
    printVector(u, (n * n));
    free(u);
    return 0;
}

float calculateError(float* old_val, float* new_val, int dim) {
    float temp_glob = 0.0;
    float temp_loc = 0.0;
    for(int i = 0; i < dim * dim; i++) {
        temp_loc = old_val[i] - new_val[i];
        if(temp_loc < 0)
            temp_loc = -temp_loc;
        if(temp_loc > temp_glob)
            temp_glob = temp_loc;
    }
    return temp_glob;
}

__host__ __device__
float functionF(float x, float y)
{ // x and y should be in (0,1)
    return 32.0f * (x * (1.0f - x) + y * (1.0f - y));
}

float* allocateSquareMatrix(int size, int initialize, int dim_n)
{
    float *tmp = (float*) malloc(size * sizeof(float));

    if (initialize)
    {
        for (int i = 0; i < dim_n; i++)
        {
            for (int j = 0; j < dim_n; j++)
            {
                tmp[i + j * dim_n] = 0.0;
                if (i == j)
                    tmp[i + j * dim_n] = 4.0;

                if (i + dim_n == j || i == j + dim_n || i + 1 == j || i == j + 1)
                    tmp[i + j * dim_n] = -1.0;

                if ((i % dim_n == 0 && j == i - 1) || (i == j - 1 && j % dim_n == 0))
                    tmp[i + j * dim_n] = 0.0;
            }
        }
    }
    else
    {
        for (int i = 0; i < dim_n; i++)
        {
            for (int j = 0; j < dim_n; j++)
            {
                tmp[i + j * dim_n] = 0.0;
            }
        }
    }
    return tmp;
}

/**
 *  Only frees the "rows" of the allocated Matrix. 
 *  Still have to call free on pointer of pointers
 */
void freeSquareMatrix(float **matrix, int dim)
{
    for (int i = 0; i < dim; i++)
    {
        free(matrix[i]);
    }
}

float* allocateVector(int size, int initialize)
{
    float *tmp = (float *)malloc(size * sizeof(float));
    if (initialize)
    {
        for (int i = 0; i < size; i++)
        {
            tmp[i] = 0.0;
        }
    }
    return tmp;
}
void printSquareMatrix(float **matrix, int dim)
{
    printf("Printing sqare matrix with dim = %d\n", dim);
    for (int i = 0; i < dim; i++)
    {
        for (int j = 0; j < dim; j++)
        {
            printf("|%f", matrix[i][j]);
        }
        printf("|\n");
    }
}

void printSquareMatrix(float *matrix, int dim) {
    printf("Printing sqare matrix with dim = %d\n", dim);
    for (int i = 0; i < dim; i++)
    {
        for (int j = 0; j < dim; j++)
        {
            printf("|%f", matrix[i + j * dim]);
        }
        printf("|\n");
    }
}

void printVector(float *vector, int length)
{
    printf("Printing Vector with length = %d\n", length);
    for (int i = 0; i < length; i++)
        printf("|%f", vector[i]);

    printf("|\n");
}

void printVectorInBlock(float *vector, int length, int blockLength)
{
    printf("Printing Vector with length = %d\n", length);
    for (int i = 0; i < length / blockLength; i++)
    {
        for (int j = 0; j < blockLength; j++)
        {
            printf("|%f", vector[i + blockLength * j]);
        }
        printf("|\n");
    }
}

void checkForError(const char* msg) {
    cudaError_t error = cudaGetLastError();
    if(error != cudaSuccess)
        printf("ERROR: %s: %s\n", msg, cudaGetErrorString(error));
}