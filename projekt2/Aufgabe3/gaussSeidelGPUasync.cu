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

//const int ITERATE_ON_BLACK = 0;
//const int ITERATE_ON_RED = 1;
//const int THREADS_PER_BLOCK = 32;
const int BLOCK_DIMENSION = 8;

__global__ void blockIterationAsyncJacobi(int dim_u, int dim_u_emb, float h, float* u_emb, int iterCount, float global_error) {


    __shared__ float localBlock[BLOCK_DIMENSION + 2][BLOCK_DIMENSION + 2];
    global_error = FEHLERSCHRANKE + 1;
    int global_block_start_i = (blockIdx.x * BLOCK_DIMENSION) + 1;
    int global_block_start_j = (blockIdx.y * BLOCK_DIMENSION) + 1;

    int global_i = 0;
    int global_j = 0;
    int local_i = 0;
    int local_j = 0;

    for(int iterations = 0; iterations < iterCount; iterations++) {

        // 1. Load data from global memory into local
        // load all needed data from global memory
        // 8x8 inner with neighbor elements around
        for(int i = threadIdx.x; i < (BLOCK_DIMENSION + 2) * (BLOCK_DIMENSION + 2); i += blockDim.x) {

            local_i = i % (BLOCK_DIMENSION + 2);
            local_j = (int) ( i / (BLOCK_DIMENSION + 2));
            global_i = global_block_start_i + local_i - 1;
            global_j = global_block_start_j + local_j - 1;

            if(global_i < dim_u_emb && global_j < dim_u_emb) {
                #ifdef PRINT
                if(blockIdx.x == 1 && blockIdx.y == 1) {
                    //printf("LOAD - ThreadIdx = %d, block.x/y = %d/%d i = %d loc_(i, j) = (%d, %d), glo_(i, j) = (%d, %d)\n", threadIdx.x, global_block_start_i, global_block_start_j, i, local_i, local_j, global_i, global_j);
                }
                //printf("ThreadIdx = %d, block.x/y = %d/%d i = %d, local_i = %d, local_j = %d, global_i = %d, global_j = %d\n", threadIdx.x, blockIdx.x, blockIdx.y, i, local_i, local_j, global_i, global_j);
                #endif
                localBlock[local_i][local_j] = u_emb[global_i + global_j * dim_u_emb];
            }
        }
        __syncthreads();

        #ifdef PRINT

        #endif

        // 2. do jacobi iteration on local block
        int i, j;
        
        for(int iter_flag = 0; iter_flag < 2; iter_flag++) {

            int threadID = threadIdx.x * 2 + iter_flag;

            // calc the index, a thread has to work on
            // only in inner block, so add +1 to avoid iteration on neighbour elements
            j = ((int) threadID / BLOCK_DIMENSION) + 1;
            i = (((int) threadID % BLOCK_DIMENSION) + (1 - 2 * iter_flag) * (j % 2) ) + 1;
            global_i = global_block_start_i + i;
            global_j = global_block_start_j + j;        

            if(global_i < dim_u_emb && global_j < dim_u_emb) {

                #ifdef PRINT
                //printf("ThreadID = %d|x=%d|y=%d, i = %d, j = %d\n", threadID, blockIdx.x, blockIdx.y, i, j);
                #endif

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
                float newU = (h * h * functionF(global_i * h, global_j * h) + tempSum) / 4.0;
                // 4. replace old value
                localBlock[i][j] = newU;
            }
        }

        // 3. write back to global memory
        // only updated values in embedded 8x8 block
        for(int i = threadIdx.x; i < BLOCK_DIMENSION * BLOCK_DIMENSION; i += blockDim.x) {

            local_i = (i % BLOCK_DIMENSION) + 1;
            local_j = (int) ( i / BLOCK_DIMENSION) + 1;
            global_i = global_block_start_i + local_i - 1;
            global_j = global_block_start_j + local_j - 1;

            if(global_i < dim_u_emb && global_j <= dim_u_emb) {
                #ifdef PRINT
                if(blockIdx.x == 0 && blockIdx.y == 0) {
                    //printf("ThreadIdx = %d, block.x/y = %d/%d i = %d loc_(i, j) = (%d, %d), glo_(i, j) = (%d, %d)\n", threadIdx.x, global_block_start_i, global_block_start_j, i, local_i, local_j, global_i, global_j);
                }
                #endif
                u_emb[global_i + global_j * dim_u_emb] = localBlock[local_i][local_j];
            }
        }
        __syncthreads();
    }
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
    int threadsPerBlock = (int) (BLOCK_DIMENSION * BLOCK_DIMENSION) / 2 + (((BLOCK_DIMENSION * BLOCK_DIMENSION) % 2) ? 2 : 0);
    dim3 numBlocks(blocksPerDimension, blocksPerDimension);

    printf("Running with numBlocks: %d, %d - %d Threads / Block.\n", blocksPerDimension, blocksPerDimension, threadsPerBlock);
    // Iterate as long as we do not come below our fehlerSchranke

    int count = 0;
    int next_err_check = 1;
    int step_and_error_arr[2][2];
    int block_iter = 1;

    while (fehlerSchranke < fehler)
    {

        blockIterationAsyncJacobi<<<numBlocks, threadsPerBlock>>>(n, n_emb, h, gpu_u_emb, block_iter, global_error);

        if(count > 1) {
            checkForError("Some shit happend.");
            cudaDeviceSynchronize();
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
        if(count > 1) {
            break;
        }
        #endif
    }

    printf("Took at least %d Block-Iterations per Warp.\n", count);
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

    printVectorInBlock(u, (n * n), n);
    //printVector(u, (n * n));
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
