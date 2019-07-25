#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <time.h>

#define FEHLERSCHRANKE 0.000001
//Exponent der Verfeinerung

float functionF(float x, float y);
float* allocateSquareMatrix(int size, int initialize, int n);
float* allocateVector(int size, int initialize);
void printSquareMatrix(float **matrix, int dim);
void printSquareMatrix(float *matrix, int dim);
void printVector(float *vector, int length);
void freeSquareMatrix(float **matrix, int dim);
void calculateError(float *old_val, float *new_val, int dim, float *result);
void printVectorInBlock(float *vector, int length, int blockLength);
void checkForError(const char* msg);

const int ITERATE_ON_BLACK = 0;
const int ITERATE_ON_RED = 1;
const int THREADS_PER_BLOCK = 32;
const int BLOCK_DIMENSION = 8;

__global__ void blockIterationAsyncJacobi(int dim_u, int dim_u_emb, float h, float* u_emb, int iterCount) {

    // 1. Load data from global memory into local
    // load all needed data from global memory
    // change only values in inner block and 
    // write only updated values to global Memory
    __shared__ float localBlock[BLOCK_DIMENSION + 2][BLOCK_DIMENSION + 2];

    int global_start_i = blockIdx.x * BLOCK_DIMENSION;
    int global_start_j = blockIdx.y * BLOCK_DIMENSION;
    int global_end_i = (blockIdx.x + 1) * BLOCK_DIMENSION + 1;
    int global_end_j = (blockIdx.y + 1) * BLOCK_DIMENSION + 1;

    int global_i = 0;
    int global_j = 0;
    int local_i = 0;
    int local_j = 0;



    for(int i = threadIdx.x; i < (BLOCK_DIMENSION + 2) * (BLOCK_DIMENSION + 2); i+=blockDim.x) {

        local_i = i % BLOCK_DIMENSION;
        local_j = (int) ( i / BLOCK_DIMENSION);
        global_i = global_start_i + local_i;
        global_j = global_start_j + local_j;

        #ifdef PRINT
        printf("ThreadIdx = %d, i = %d, local_i = %d, local_j = %d, global_i = %d, global_j = %d", threadIdx.x, i, local_i, local_j, global_i, global_j);
        #endif

        if(global_i > 0 && global_i < dim_u_emb && global_j > 0 && )
            localBlock[local_i][local_j] = u_emb[global_i + global_j * dim_u_emb];
    }
    __syncthreads();

    /*
    // 2. do jacobi iterations on local block
    for(int iterations = 0; iterations < iterCount; iterations++) {

        for(int iter_flag = 0; iter_flag < 2; iter_flag++) {

            int threadID = threadIdx.x * 2 + iter_flag;
            // use index of thread to calculate position in matrix 
            // to execute computation on
            int i, j;
        
            // 1. Calculate the index's in the matrix a thread has to work on
            j = ((int) threadID / BLOCK_DIMENSION) + 1;
            i = ((int) threadID % BLOCK_DIMENSION + (1 - 2 * iter_flag) * (j % 2) ) + 1;
        
            if(i > 0 && i < BLOCK_DIMENSION + 1 && j < BLOCK_DIMENSION - 1 && j > 0) {

                #ifdef PRINT
                printf("ThreadID = %d, i = %d, j = %d\n", threadID, i, j);
                #endif
        
                // 3. calculate new value for u_emb
                float tempSum = 
                    // top element
                    localBlock[i][j - 1] 
                    // left element
                    + localBlock[i - 1][j] 
                    // right element
                    + localBlock[i + 1][j] 
                    // bottom element
                    + localBlock[i][j + 1]; 

                // calc new value for u
                float newU = (h * h * functionF((j_inner / dim_u + 1) * h, (j_inner % dim_u + 1) * h) + tempSum) / 4.0;
                // 4. replace old value
                
                localBlock[i][j] = newU;
            }
        }
        __syncthreads();
    }

    cudaThreadSynchronize();
    */
}


/**
*   n: Dimension of u matrix
*   fehlerSchranke: Abbruchbedingung für Gaus Seidel Verfahren
*   h: Verfeinerung/Gitterschrittweite
*   a: pointer to 2D-Array of Matrix a (currently not used, due 
*       to extraction of calculation pattern into algorithm)
*   u: pointer to u matrix
*/
void jaccobi(int n, float fehlerSchranke, float h, float *u)
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
    printSquareMatrix(u_emb_new, n_emb);
#endif

    // allocate device memory
    float *gpu_u_emb;
    cudaMalloc((void**)&gpu_u_emb, n_emb * n_emb * sizeof(float));
    // copy from host to device
    cudaMemcpy(gpu_u_emb, u_emb, n_emb * n_emb * sizeof(float), cudaMemcpyHostToDevice);
    
    // calculate the blocks per dimension
    int blocksPerDimension = n_emb / BLOCK_DIMENSION + (n_emb % BLOCK_DIMENSION ? 1: 0);
    int threadsPerBlock = (int) (BLOCK_DIMENSION * BLOCK_DIMENSION) / 2 + (((BLOCK_DIMENSION * BLOCK_DIMENSION) % 2) ? 2 : 0) ;
    dim3 numBlocks(blocksPerDimension, blocksPerDimension);

    printf("Running with numBlocks: %d, %d - %d Threads / Block.\n", blocksPerDimension, blocksPerDimension, threadsPerBlock);
    // Iterate as long as we do not come below our fehlerSchranke
    int count = 0;
    int block_iter = 5;
    while (fehlerSchranke < fehler)
    {
        // black iteration
        //redBlackIteration<<<numBlocks, THREADS_PER_BLOCK>>>(n, n_emb, h, gpu_u_emb, ITERATE_ON_BLACK);
        //cudaDeviceSynchronize();
        // red iteration
        //redBlackIteration<<<numBlocks, THREADS_PER_BLOCK>>>(n, n_emb, h, gpu_u_emb, ITERATE_ON_RED);
        //cudaDeviceSynchronize();
        blockIterationAsyncRedBlack<<<numBlocks, threadsPerBlock>>>(n, n_emb, h, gpu_u_emb, block_iter);

        if(count > 1) {
            // move result of first iteration onto host (implicitly synchronizing)
            cudaMemcpy(u_emb_new, gpu_u_emb, n_emb * n_emb * sizeof(float), cudaMemcpyDeviceToHost);
            
            // calculate error
            calculateError(u_emb, u_emb_new, n_emb, &fehler);
            // switch pointers
            float *temp = u_emb;
            u_emb = u_emb_new;
            u_emb_new = temp;
        }

        
        // count for iterations
        count += block_iter;

        #ifdef PRINT
        printf("Iteration-Error = %.8f\n", fehler);
        printSquareMatrix(u_emb_new, n_emb);
        break;
        #endif
    }
    printf("Took %d Iterations to complete.\n", count);
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
    cudaFree(gpu_u_emb);
    free(u_emb_new);
    free(u_emb);
}

int main()
{

    clock_t start, stop;
    double time_used;

    start = clock();
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
    printVector(u, (n * n));
#endif

    
    // executing gauss seidel verfahren
    jaccobi(n, FEHLERSCHRANKE, h, u);

    stop = clock();
    time_used = (double) (stop - start) / CLOCKS_PER_SEC;

    printf("Time used %f\n", time_used);

    //printVectorInBlock(u, (n * n), n);
    printVector(u, (n * n));
    free(u);

    return 0;
}

/**
 * Calculating distance between two vectors via L2-Norm  
 */
void calculateError(float* old_val, float* new_val, int dim, float *result) {

    float sum = 0.0;
    #pragma omp parallel for shared(old_val, new_val) reduction(+: sum)
    for(int i = 0; i < dim * dim; i++) {
        sum += (new_val[i] - old_val[i]) * (new_val[i] - old_val[i]);
    }
    *result = sqrtf(sum);
    //*result = temp_glob;
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
            printf(" %f", matrix[i][j]);
        }
        printf(" \n");
    }
}

void printSquareMatrix(float *matrix, int dim) {
    printf("Printing sqare matrix with dim = %d\n", dim);
    for (int i = 0; i < dim; i++)
    {
        for (int j = 0; j < dim; j++)
        {
            printf(" %f", matrix[i + j * dim]);
        }
        printf(" \n");
    }
}

void printVector(float *vector, int length)
{
    printf("Printing Vector with length = %d\n", length);
    for (int i = 0; i < length; i++)
        printf(" %f", vector[i]);

    printf(" \n");
}

void printVectorInBlock(float *vector, int length, int blockLength)
{
    printf("Printing Vector with length = %d\n", length);
    for (int i = 0; i < length / blockLength; i++)
    {
        for (int j = 0; j < blockLength; j++)
        {
            printf(" %f", vector[i + blockLength * j]);
        }
        printf(" \n");
    }
}

void checkForError(const char* msg) {
    cudaError_t error = cudaGetLastError();
    if(error != cudaSuccess)
        printf("ERROR: %s: %s\n", msg, cudaGetErrorString(error));
}
