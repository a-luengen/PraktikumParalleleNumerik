#include <stdio.h>
#include <omp.h>
#include <stdlib.h>
#include <math.h>
#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

#define FEHLERSCHRANKE 0.0001
#define LVALUE 3
#define M 3
#define N 2

struct Matrix
{
  int rows;
  int cols;
  float **data;
};

struct Vector
{
  int size;
  float *data;
};

float functionF(float x, float y);
float norm(struct Vector v);
float scalar(struct Vector v1, struct Vector v2);
struct Vector multMatrixVector(struct Matrix m, struct Vector v);
struct Vector diffVectors(struct Vector v1, struct Vector v2);
struct Vector sumVectors(struct Vector v1, struct Vector v2);
struct Vector multFloatVector(float factor, struct Vector v);
struct Matrix transpose(struct Matrix m);

int main()
{


  float h = 1.0;
  int n = 1;
  for (int i = 0; i < LVALUE; i++)
  {
    h = h / 2.0;
    n = n * 2;
  }
  n = n - 1 +2;
  float timeStepWidth = h;
  int timeStepIterations = 7.0f/timeStepWidth;
  struct Vector heat;
  heat.size = n * n;
  heat.data = malloc(n * n * sizeof(float));
  #pragma omp parallel for
  for (int i = 0; i < n * n; i++)
  { //set heat
    heat.data[i] = 20.0f+timeStepWidth*functionF((i/n)*h,(i%n)*h);
  }

      //Form the solution
    struct Vector solution;
      solution.size = n * n;
      solution.data = malloc(n * n * sizeof(float));
      #pragma omp parallel for
      for (int i = 0; i < n * n; i++)
      { //set heat
        solution.data[i] = 0.0f;
      }
  struct Vector heatLast;
  heatLast.size = n * n;
  heatLast.data = malloc(n * n * sizeof(float));
  #pragma omp parallel for
  for (int i = 0; i < n * n; i++)
  { //set heat
    heatLast.data[i] = 20.0f;
  }



  //Matrix A
  struct Matrix A;
  A.rows = n * n;
  A.cols = n * n;
  A.data = malloc(n * n * n * n * sizeof(float));

  //A besetzen
  #pragma omp parallel for
  for (int i = 0; i < n * n; i++)
  {
    A.data[i] = malloc(n * n * sizeof(float));
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
  struct Matrix U;
  U.rows = n * n;
  U.cols = n * n;
  U.data = malloc(n * n * n * n * sizeof(float));

  //U besetzen
  #pragma omp parallel for
  for (int i = 0; i < n * n; i++)
  {
    U.data[i] = malloc(n * n * sizeof(float));
    for (int j = 0; j < n * n; j++)
    {
      U.data[i][j] = 0.0f;
    }
  }

  //Matrix L
  struct Matrix L;
  L.rows = n * n;
  L.cols = n * n;
  L.data = malloc(n * n * n * n * sizeof(float));

  //L besetzen
  #pragma omp parallel for
  for (int i = 0; i < n * n; i++)
  {
    L.data[i] = malloc(n * n * sizeof(float));
    for (int j = 0; j < n * n; j++)
    {
      L.data[i][j] = 0.0f;
    }
  }

  float testSum = 1.0f;

  //compute preconditioner
  while(testSum > 0.00001f){ // until convergence!
    testSum = 0.0f;

    #pragma omp parallel for schedule(static,1) reduction(+:testSum)
    for(int i = 0; i < n*n; i++){
      for (int j = i; j <= i+n && j < n*n; j++){
        float sm = A.data[i][j];
        if(sm != 0)
        //sum to diff
        for(int zaehler = 1; zaehler< i;zaehler++){
          sm -= U.data[zaehler][i]*U.data[zaehler][j];
        }
        if(i != j){
          U.data[i][j]=sm/U.data[i][i];
          float helperSum= L.data[j][i] - U.data[i][j];
          if(helperSum < 0){helperSum = helperSum * -1.0f;}
          testSum = helperSum;
          L.data[j][i] = U.data[i][j];
        } else {
          U.data[i][i] = sqrt(sm);
          float helperSum= L.data[j][i] - U.data[i][j];
          if(helperSum < 0){helperSum = helperSum * -1.0f;}
          testSum = helperSum;
          L.data[i][i] = U.data[i][i];
        }
      }
    }
  }

  int k = 1000;
  //Matrix H
  struct Matrix H;
  H.rows = k + 1;
  H.cols = k;
  H.data = malloc(H.rows * H.cols * sizeof(float));
  //H besetzen
  #pragma omp parallel for
  for (int i = 0; i < k + 1; i++)
  {
    H.data[i] = malloc((k) * sizeof(float));
    for (int j = 0; j < k; j++)
    {
      H.data[i][j] = 0.0f;
    }
  }

  for(int timeCounter = 0; timeCounter < timeStepIterations;timeCounter++){
    k = 2000;

    //Lösungsvektoren x0
    struct Vector x0;
    x0.size = n * n;
    x0.data = malloc(n * n * sizeof(float));
    #pragma omp parallel for
    for (int i = 0; i < n * n; i++)
    { //set x0
      x0.data[i] = solution.data[i];
    }

    //Funktionsvektor b
    struct Vector b;
    b.size = n * n;
    b.data = malloc(n * n * sizeof(float));
    #pragma omp parallel for
    for (int i = 0; i < n; i++)
    {
      for (int j = 0; j < n; j++)
      {
        b.data[i * n + j] = h*h*(functionF(i*h,j*h)-(heatLast.data[i*n+j]-heat.data[i*n+j])/timeStepWidth);
      }
    }



    //Vectoren Vk
    struct Vector *V;
    V = malloc((k + 1) * n * n * sizeof(float) + (k + 1) * sizeof(int));
    //Vk besetzen
    #pragma omp parallel for
    for (int i = 0; i < k + 1; i++)
    {
      V[i].size = n * n;
    }

    struct Vector c;
    c.size = k + 1;
    c.data = malloc((k + 1) * sizeof(float));
    #pragma omp parallel for
    for (int i = 0; i < k + 1; i++)
    {
      c.data[i] = 0.0f;
    }

    struct Vector w;
    w.size = n*n;
    w.data = malloc((n*n) * sizeof(float));
    #pragma omp parallel for
    for (int i = 0; i < n *n; i++)
    {
      w.data[i] = 0.0f;
    }

    struct Vector gamma;
    gamma.size = k + 1;
    gamma.data = malloc((k + 1) * sizeof(float));
    #pragma omp parallel for
    for (int i = 0; i < k + 1; i++)
    {
      gamma.data[i] = 0.0f;
    }
    struct Vector s;
    s.size = k + 1;
    s.data = malloc((k + 1) * sizeof(float));
    #pragma omp parallel for
    for (int i = 0; i < k + 1; i++)
    {
      s.data[i] = 0.0f;
    }

    struct Vector r0;
    r0.size = n*n;
    r0.data = malloc(n*n * sizeof(float));
    #pragma omp parallel for
    for (int i = 0; i < n*n; i++)
    {
      r0.data[i] = 0.0f;
    }

    //compute r0
    r0 = diffVectors(b, multMatrixVector(A, x0));
    //berechne preconditioned r0
    for (int i = 0; i < n*n; i++)
    { //Muss in reihenfolge laufen
      if(i<n||i%n == 0 ||i%n==n-1||i>n*(n-1))
      r0.data[i]=0;
      else {
        float helperValue = r0.data[i];
        for (int l = i - 1; l >=l-n &&l >= 0; l--)
        {
          helperValue -= L.data[i][l] * r0.data[l];
        }
        r0.data[i] = helperValue / L.data[i][i];
      }
    }
    for (int i = n*n-1; i >=0; i--)
    { //Muss in reihenfolge laufen
      if(i<n||i%n == 0 ||i%n==n-1||i>n*(n-1))
      r0.data[i]=0;
      else {
        float helperValue = r0.data[i];
        for (int l = i + 1; l<=l+n&&l < n*n; l++)
        {
          helperValue -= U.data[i][l] * r0.data[l];
        }
        r0.data[i] = helperValue / U.data[i][i];
      }
    }

    gamma.data[0] = norm(r0);
    V[0] = multFloatVector(1.0f / gamma.data[0], r0);

    for (int j = 0; j < k; j++)
    {
      struct Vector q = multMatrixVector(A, V[j]);
      for(int i = 0; i < n*n; i++){
        if(i<n||i%n == 0 ||i%n==n-1||i>n*(n-1))
        q.data[i]=0;
      }
      //berechne preconditioned w
      for (int i = 0; i < n*n; i++)
      { //Muss in reihenfolge laufen
        if(i<n||i%n == 0 ||i%n==n-1||i>n*(n-1))
        w.data[i]=0;
        else {
          float helperValue = q.data[i];
          for (int l = i - 1; l>=l-n&&l >= 0; l--)
          {
            helperValue -= L.data[i][l] * w.data[l];
          }
          w.data[i] = helperValue / L.data[i][i];
        }
      }
      for (int i = n*n-1; i >=0; i--)
      { //Muss in reihenfolge laufen
        if(i<n||i%n == 0 ||i%n==n-1||i>n*(n-1))
        w.data[i]=0;
        else {
          float helperValue = w.data[i];
          for (int l = i + 1; l<=l+n&&l < n*n; l++)
          {
            helperValue -= U.data[i][l] * w.data[l];
          }
          w.data[i] = helperValue / U.data[i][i];
        }
      }
      //hij berechnen
      #pragma omp parallel for
      for (int i = 0; i <= j; i++)
      {
        H.data[i][j] = scalar(w, V[i]);
      }
      //unnormedVj+1 berechnen
      struct Vector summedVector;
      summedVector.size = n * n;
      summedVector.data = malloc(n * n * sizeof(float));
      #pragma omp parallel for
      for (int zaehler = 0; zaehler < n * n; zaehler++)
      {
        summedVector.data[zaehler] = 0.0f;
      }
      for (int i = 0; i <= j; i++)
      { // berechne summe zum abziehen
        struct Vector vectorToSum=multFloatVector(H.data[i][j], V[i]);
        summedVector = sumVectors(summedVector, vectorToSum);
        free(vectorToSum.data);
      }
      struct Vector unnormedV;
      unnormedV = diffVectors(w, summedVector);
      for(int i = 0; i < n*n; i++){
        if(i<n||i%n == 0 ||i%n==n-1||i>n*(n-1))
        unnormedV.data[i]=0;
      }
      //setze hj+1j
      H.data[j + 1][j] = norm(unnormedV);

      //Hij Schleife
      for (int i = 0; i < j; i++)
      { //muss in reihenfolge ablaufen
        float oldhij = H.data[i][j];
        float oldhi1j = H.data[i + 1][j];
        H.data[i][j] = c.data[i + 1] * oldhij + s.data[i + 1] * oldhi1j;
        H.data[i + 1][j] = -1.0f * s.data[i + 1] * oldhij + c.data[i + 1] * oldhi1j;
      }
      float beta = sqrt(H.data[j][j] * H.data[j][j] + H.data[j + 1][j] * H.data[j + 1][j]);
      s.data[j + 1] = H.data[j + 1][j] / beta;
      c.data[j + 1] = H.data[j][j] / beta;
      H.data[j][j] = beta;
      gamma.data[j + 1] = -1.0f * s.data[j + 1] * gamma.data[j];
      gamma.data[j] = c.data[j + 1] * gamma.data[j];

      //free data
      free(summedVector.data);
      free(q.data);

      float testValue = gamma.data[j + 1];
      if (testValue < 0)
      testValue = testValue * -1.0f;

      if (testValue < FEHLERSCHRANKE)
      {
        k = j;
      }
      else
      {
        //setze vj+1
        V[j + 1] = multFloatVector(1 / H.data[j + 1][j], unnormedV);
      }

      free(unnormedV.data);
    }

    //Deklariere y
    struct Vector y;
    y.size = (k + 1);
    y.data = malloc(sizeof(float) * (k + 1));
    #pragma omp parallel for
    for (int i = 0; i < k + 1; i++)
    {
      y.data[i] = 0.0f;
    }

    //berechne y
    for (int i = k; i >= 0; i--)
    { //Muss in reihenfolge laufen
      float helperValue = gamma.data[i];
      for (int l = i + 1; l <= k; l++)
      {
        helperValue -= H.data[i][l] * y.data[l];
      }
      y.data[i] = helperValue / H.data[i][i];
    }

    //summed Vector of Vi and yi
    struct Vector summedVector;
    summedVector.size = n * n;
    summedVector.data = malloc(n * n * sizeof(float));
    #pragma omp parallel for
    for (int zaehler = 0; zaehler < n * n; zaehler++)
    {
      summedVector.data[zaehler] = 0.0f;
    }
    for (int i = 0; i <= k; i++)
    { // berechne summe der Vektoren
      summedVector = sumVectors(summedVector, multFloatVector(y.data[i], V[i]));
    }
    //solution
    solution = sumVectors(x0, summedVector);
    struct Vector test = diffVectors(b, multMatrixVector(A, solution));
    //AUSGABE
    int eachXthIterationPrint = 1;
    if(timeCounter%eachXthIterationPrint == 0){
    printf("Time: %f\n",timeCounter *timeStepWidth);
  }
    for (int i = 0; i < n; i++)
    {
      for (int j = n - 1; j >=0; j--)
      {
        int indexToPrint = i + j * n;
        if(!(indexToPrint<n||indexToPrint%n == 0 ||indexToPrint%n==n-1||indexToPrint>n*(n-1))){
          float laPlaceU = solution.data[indexToPrint+1]+solution.data[indexToPrint-1]+solution.data[indexToPrint+n]+solution.data[indexToPrint-n]-4.0f*solution.data[indexToPrint];
          float uStrichNow = functionF(h * i , h * j )+laPlaceU;
          heatLast.data[indexToPrint] = heat.data[indexToPrint];
          heat.data[indexToPrint] +=timeStepWidth*uStrichNow;
 } else {
          heat.data[indexToPrint] = 20.0f;
        }
        if(timeCounter%eachXthIterationPrint == 0){
          float toPrint = heat.data[indexToPrint];
          if (toPrint < 0)
          printf("%f ", toPrint);
          else
          printf(" %f ", toPrint);
        }
      }
      if(timeCounter%eachXthIterationPrint == 0){
      printf("\n");
    }
    }
    free(test.data);
    free(x0.data);
    free(r0.data);
    free(b.data);
    free(w.data);
    free(solution.data);
  }
  //free data
  for (int i = 0; i < n * n; i++)
  {
    free(A.data[i]);
    free(L.data[i]);
    free(U.data[i]);
  }
  free(A.data);
  free(U.data);
  free(L.data);


  return 0;
}

float functionF(float x, float y)
{ // x and y should be in (0,1)
  return 32.0f*(x*(1.0f-x) + y*(1.0f-y));//16.0f*(x*(1.0f-x) * y*(1.0f-y));//
}

struct Vector multMatrixVector(struct Matrix m, struct Vector v)
{
  struct Vector toReturn;
  toReturn.size = m.rows;
  toReturn.data = malloc(sizeof(float) * m.rows);
  #pragma omp parallel for
  for (int i = 0; i < m.rows; i++)
  {
    toReturn.data[i] = 0.0f;
    for (int j = 0; j < m.cols; j++)
    {
      toReturn.data[i] += m.data[i][j] * v.data[j];
    }
  }
  return toReturn;
}

struct Vector diffVectors(struct Vector v1, struct Vector v2)
{
  struct Vector toReturn;
  toReturn.size = v1.size;
  toReturn.data = malloc(sizeof(float) * v1.size);
  #pragma omp parallel for
  for (int i = 0; i < v1.size; i++)
  {
    toReturn.data[i] = v1.data[i] - v2.data[i];
  }
  return toReturn;
}

struct Vector multFloatVector(float factor, struct Vector v)
{
  struct Vector toReturn;
  toReturn.size = v.size;
  toReturn.data = malloc(sizeof(float) * v.size);
  #pragma omp parallel for
  for (int i = 0; i < v.size; i++)
  {
    toReturn.data[i] = factor * v.data[i];
  }
  return toReturn;
}

struct Vector sumVectors(struct Vector v1, struct Vector v2)
{
  struct Vector toReturn;
  toReturn.size = v1.size;
  toReturn.data = malloc(sizeof(float) * v1.size);
  #pragma omp parallel for
  for (int i = 0; i < v1.size; i++)
  {
    toReturn.data[i] = v1.data[i] + v2.data[i];
  }
  return toReturn;
}

float scalar(struct Vector v1, struct Vector v2)
{
  float sum = 0.0f;
  for (int i = 0; i < v1.size; i++)
  {
    sum += v1.data[i] * v2.data[i];
  }
  return sum;
}

float norm(struct Vector v)
{
  return sqrt(scalar(v, v));
}
