#include <stdio.h>
#include <omp.h>
#include <stdlib.h>
#include <math.h>

#define FEHLERSCHRANKE 1.0
#define L 6
#define M 1
#define N 2

struct Matrix {
  int rows;
  int cols;
  float** data;
};

struct Vector {
  int size;
  float* data;
};

float functionF(float x, float y);
float norm(struct Vector v);
float scalar(struct Vector v1, struct Vector v2);
struct Vector multMatrixVector(struct Matrix m, struct Vector v);
struct Vector diffVectors(struct Vector v1, struct Vector v2);
struct Vector sumVectors(struct Vector v1, struct Vector v2);
struct Vector multFloatVector(float factor, struct Vector v);
struct Matrix transpose(struct Matrix m);

int main() {
 int k = 5;
 float h = 1.0;
 int n = 1;
 for(int i = 0; i < L; i++){
   h = h/2.0;
   n = n*2;
 }
 n = n-1;

 //Lösungsvektoren x0
 struct Vector x0;
 x0.size = n*n;
 x0.data = malloc(n*n*sizeof(float));
 for(int i = 0; i < n*n; i++){ //set x0
   x0.data[i] = 0.0;
 }

 //Funktionsvektor b
 struct Vector b;
 b.size = n*n;
 b.data = malloc(n*n*sizeof(float));
 for(int i= 0; i< n; i++) {
   for(int j = 0; j < n; j++) {
     b.data[i*n +j] = h*h*functionF(h*i,h*j);
   }
 }

 //Matrix A
 struct Matrix A;
 A.rows = n*n;
 A.cols = n*n;
 A.data = malloc(A.rows*A.cols*sizeof(float));

 //A besetzen
 for(int i = 0; i < n*n ;i++){
   A.data[i] = malloc(n*n*sizeof(float));
   for(int j = 0; j< n*n; j++){
     A.data[i][j] = 0.0;
     if(i == j)
       A.data[i][j] = 4.0;

     if(i+n == j || i == j+n|| i+1 == j || i == j+1)
       A.data[i][j] = -1.0;

     if((i%n == 0 && j == i-1) || (i == j-1 && j%n == 0))
       A.data[i][j] =0.0;
   }
 }

 //Matrix H
 struct Matrix H;
 H.rows = k+1;
 H.cols = k;
 H.data = malloc(H.rows*H.cols*sizeof(float));
 //H besetzen
 for(int i = 0; i < k+1;i++){
   H.data[i] = malloc((k)*sizeof(float));
   for(int j = 0; j< k; j++){
     H.data[i][j] = 0.0f;
   }
 }

 //Vectoren Vk
 struct Vector* V;
 V = malloc((k+1)*n*n*sizeof(float) + (k+1)*sizeof(int));
 //Vk besetzen
 for(int i = 0; i < k+1; i++) {
   V[i].size = n*n;
   V[i].data =malloc(n*n*sizeof(float));
   for(int j = 0; j< n*n; j++ ){
     V[i].data[j] = 0.0f;
   }

 }

 //compute r0
 struct Vector r0 =  diffVectors(b, multMatrixVector(A,x0));
 V[0] = multFloatVector(1.0f/norm(r0),r0);

 for(int j = 0; j < k; j++) {
   //hij berechnen
   for(int i = 0; i <= j;i++){
     H.data[i][j] = scalar(multMatrixVector(A,V[j]),V[i]);
   }
   //unnormedVj+1 berechnen
   struct Vector summedVector;
   summedVector.size = n*n;
   summedVector.data =malloc(n*n*sizeof(float));
   for(int zaehler = 0; zaehler< n*n; zaehler++ ){
     summedVector.data[zaehler] = 0.0f;
   }
   for(int i= 0; i < j; i++){ // berechne summe zum abziehen
     summedVector = sumVectors(summedVector, multFloatVector(H.data[i][j], V[i]));
   }
   struct Vector unnormedV;
   unnormedV = diffVectors(multMatrixVector(A, V[j]), summedVector);

   //setze hj+1j
   H.data[j+1][j] =norm(unnormedV);
   //setze vj+1
   V[j+1] = multFloatVector(1/H.data[j+1][j], unnormedV);

   //free data
   free(unnormedV.data);
   free(summedVector.data);
 }

 //Löse Gleichungssystem Hy = (||r0||,||r0||, ...)
 struct Vector y;
 y.size = k;
 y.data = malloc(sizeof(float) * k);
 float r0Norm = norm(r0);
//FEHLERQUELLE Y DATA INIT?????????????? TODO!!
 for(int i= k; i>=1; i-- ){
   float helperValue = r0Norm;
   for(int j=i; j < k; j++){
     helperValue -= y.data[j]*H.data[i][j];
   }
   y.data[i-1] = helperValue / H.data[i][i-1];
 }

 //Form the solution
 struct Vector solution;
//summed Vector of Vi and yi
struct Vector summedVector;
summedVector.size = n*n;
summedVector.data =malloc(n*n*sizeof(float));
for(int zaehler = 0; zaehler< n*n; zaehler++ ){
  summedVector.data[zaehler] = 0.0f;
}
for(int i= 0; i < k; i++){ // berechne summe zum abziehen
  summedVector = sumVectors(summedVector, multFloatVector(y.data[i], V[i]));
}
//solution
solution = sumVectors(x0,summedVector);
struct Vector test =  diffVectors(b, multMatrixVector(A,solution));
printf("Residuum: %f \n",norm(test));
//AUSGABE
/*
 for(int i = 0; i < k+1; i++){
   for(int j = 0; j < k; j++){

     if(H.data[i][j] < 0)
       printf("%f ",H.data[i][j]);
     else
       printf(" %f ",H.data[i][j]);

   }
   printf ("\n");
 }*/
   //print solution
   printf ("\n");
   for(int i = 0; i < n; i++){
     if(i ==0){
       for(int j = 0; j <n+2; j++){
          printf(" %f ",0.0f);
       }
       printf ("\n");
     }
     for(int j = n-1; j > -1; j--){
       if(j == n-1)printf(" %f ",0.0f);

       if(solution.data[i +j*n] < 0)
        printf("%f ",solution.data[i +j*n]);
       else
        printf(" %f ",solution.data[i +j*n]);

       if(j ==0)printf(" %f ",0.0f);
     }
     printf ("\n");
     if(i ==n-1){
       for(int j = 0; j <n+2; j++){
          printf(" %f ",0.0f);
       }
       printf ("\n");
     }
  }//Schönere Ausgabe schreiben
 //free data
 for(int i = 0; i < n*n; i++){
   free(A.data[i]);
 }
 free(A.data);
 free(r0.data);
 free(b.data);
 free(x0.data);
 return 0;
}



float functionF(float x, float y) { // x and y should be in (0,1)
  return (M*M+N*N)*(4*M_PI*M_PI*sin(2*M*M_PI*x)*sin(2*N*M_PI*y));
}

struct Vector multMatrixVector(struct Matrix m, struct Vector v) {
  struct Vector toReturn;
  toReturn.size = m.rows;
  toReturn.data = malloc(sizeof(float) * m.rows);
  for(int i = 0; i< m.rows; i++){
    for(int j = 0; j < m.cols;j++){
      toReturn.data[i] += m.data[i][j] *v.data[j];
    }
  }
  return toReturn;
}

struct Vector diffVectors(struct Vector v1, struct Vector v2) {
  struct Vector toReturn;
  toReturn.size = v1.size;
  toReturn.data = malloc(sizeof(float) * v1.size);
  for(int i = 0; i< v1.size; i++){
    toReturn.data[i] = v1.data[i] - v2.data[i];
  }
  return toReturn;
}

struct Vector multFloatVector(float factor, struct Vector v) {
  struct Vector toReturn;
  toReturn.size = v.size;
  toReturn.data = malloc(sizeof(float) * v.size);
  for(int i = 0; i< v.size; i++){
    toReturn.data[i] = factor *v.data[i];
  }
  return toReturn;
}

struct Vector sumVectors(struct Vector v1, struct Vector v2) {
  struct Vector toReturn;
  toReturn.size = v1.size;
  toReturn.data = malloc(sizeof(float) * v1.size);
  for(int i = 0; i< v1.size; i++){
    toReturn.data[i] = v1.data[i] + v2.data[i];
  }
  return toReturn;
}

float scalar(struct Vector v1, struct Vector v2) {
  float sum = 0.0f;
  for(int i = 0; i< v1.size;i++){
    sum+= v1.data[i] * v2.data[i];
  }
  return sum;
}

float norm(struct Vector v) {
  return sqrt(scalar(v,v));
}

struct Matrix transpose(struct Matrix m) {
  struct Matrix transposed;
  transposed.cols = m.rows;
  transposed.rows = m.cols;
  transposed.data = malloc(sizeof(float)*m.rows*m.cols);
  for(int i = 0; i < m.cols; i++) {
    transposed.data[i] = malloc(sizeof(float) * m.cols);
    for(int j = 0; j< m.rows; j++) {
      transposed.data[i][j] = m.data[j][i];
    }
  }
  return transposed;
}
