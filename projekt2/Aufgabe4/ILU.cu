#include <stdio.h>
#include <omp.h>
#include <stdlib.h>
#include <math.h>

#define FEHLERSCHRANKE 0.00001
#define L_VALUE 2
#define M 3
#define N 2

struct Matrix
{
  int rows;
  int cols;
  float **data;
};
__global__
void computeIteration(Matrix A,Matrix U,Matrix L, int n, float* sum){
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = blockDim.x * gridDim.x;
  for(int i = index; i < n*n; i+=stride){
    for (int j = i; j < n*n&& j<=i+n; j++){
      float sm = A.data[i][j];
      //sum to diff
      for(int k = 1; k< i;k++){
        sm -= U.data[k][i]*U.data[k][j];
      }

      if(i != j){
        U.data[i][j]=sm/U.data[i][i];
        float helperSum= L.data[j][i] - U.data[i][j];
        if(helperSum < 0){helperSum = helperSum * -1.0f;}
        sum[0] = helperSum;
        L.data[j][i] = U.data[i][j];
      } else {
        U.data[i][i] = sqrt(sm);
        float helperSum= L.data[i][i] - U.data[i][i];
        if(helperSum < 0){helperSum = helperSum * -1.0f;}
        sum[0] = helperSum;
        L.data[i][i] = U.data[i][i];
      }
    }
  }
}
int main()
{
  float h = 1.0;
  int n = 1;
  for (int i = 0; i < L_VALUE; i++)
  {
    h = h / 2.0;
    n = n * 2;
  }
  n = n - 1;

  //Matrix A
  Matrix A;
  A.rows = n * n;
  A.cols = n * n;
  cudaMallocManaged(&A.data, A.rows * A.cols * sizeof(float));
  //A besetzen
  for (int i = 0; i < n * n; i++)
  {
    cudaMallocManaged(&A.data[i], n*n * sizeof(float));
    for (int j = 0; j < n * n; j++)
    {
      A.data[i][j] = 0.0;
      if (i == j)
      A.data[i][j] = 4.0;

      if (i + n == j || i == j + n || i + 1 == j || i == j + 1)
      A.data[i][j] = -1.0;

      if ((i % n == 0 && j == i - 1) || (i == j - 1 && j % n == 0))
      A.data[i][j] = 0.0;
    }
  }

  //Matrix U
  Matrix U;
  U.rows = n * n;
  U.cols = n * n;
  cudaMallocManaged(&U.data, U.rows * U.cols * sizeof(float));

  //U initialisieren
  for (int i = 0; i < n * n; i++)
  {
    cudaMallocManaged(&U.data[i], n*n * sizeof(float));
    for (int j = 0; j < n * n; j++)
    {
      U.data[i][j] = 0.0f;
    }
  }

  //Matrix L
  Matrix L;
  L.rows = n * n;
  L.cols = n * n;
  cudaMallocManaged(&L.data, L.rows * L.cols * sizeof(float));

  //L initialisieren
  for (int i = 0; i < n * n; i++)
  {
    cudaMallocManaged(&L.data[i], n*n * sizeof(float));
    for (int j = 0; j < n * n; j++)
    {
      L.data[i][j] = 0.0f;
    }
  }

  float* testSum;
  cudaMallocManaged(&testSum, sizeof(float));
  testSum[0] = 1.0f;
  while(testSum[0] > 0.00001f){ // until convergence!
    testSum[0] = 0.0f;
    computeIteration<<<176,16>>>(A,U,L,n,testSum);
    cudaDeviceSynchronize();
  }
  for(int i = 0 ; i < n*n; i++){
    printf("{");
    for(int j = 0; j < n*n; j++){
      printf("%f ", L.data[i][j]);
    }
    printf("} \n");
  }
  printf(" \n");
  for(int i = 0 ; i < n*n; i++){
    printf("{");
    for(int j = 0; j < n*n; j++){
      printf("%f ", U.data[i][j]);
    }
    printf("} \n");
  }
  printf(" \n");
  for(int i = 0 ; i < n*n; i++){
    printf("{");
    for(int j = 0; j < n*n; j++){
      float toPrint = 0;
      for(int k = 0; k < n*n;k++){
        toPrint += L.data[i][k]*U.data[k][j];
      }
      printf("%f ", toPrint);
    }
    printf("} \n");
  }

  //cuda free!!
  cudaFree(testSum);
  cudaFree(L.data);
  cudaFree(U.data);
  cudaFree(A.data);
  return 0;
}
