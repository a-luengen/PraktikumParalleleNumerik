#include <stdio.h>
#include <omp.h>
#include <stdlib.h>
#include <math.h>

#define L 6
#define FEHLERSCHRANKE 0.00001
#define M 3
#define N 2
//Exponent der Verfeinerung

float functionF(float x, float y);

int main() {
  //Randbedingungen
  float h = 1.0;
  int n = 1;
  for(int i = 0; i < L; i++){
    h = h/2.0;
    n = n*2;
  }
  n = n-1;

  //Lösungsvektoren u
  float *u = malloc(n*n*sizeof(float));
  for(int i = 0; i < n*n; i++){ //set u0
    u[i] = 0.0;
  }

  //Matrix A
	float **a = malloc(n*n*n*n*sizeof(float));

  //A besetzen
  for(int i = 0; i < n*n ;i++){
    a[i] = malloc(n*n*sizeof(float));
    for(int j = 0; j< n*n; j++){
      a[i][j] = 0.0;
      if(i == j)
        a[i][j] = 4.0;

      if(i+n == j || i == j+n|| i+1 == j || i == j+1)
        a[i][j] = -1.0;

      if((i%n == 0 && j == i-1) || (i == j-1 && j%n == 0))
        a[i][j] =0.0;
    }
  }
	//TODO: Timeranfang

  float fehler = FEHLERSCHRANKE +1;
  while(FEHLERSCHRANKE < fehler){
    for(int j = 0; j < n*n; j++) {
        float firstSum = 0.0;
        float secondSum = 0.0;
        #pragma omp parallel for shared(a,u) reduction(+:firstSum)
        for(int i = 0; i < j; i++){
          firstSum += a[j][i] * u[i];
        }
        #pragma omp parallel for shared(a,u) reduction(+:secondSum)
        for(int i = j+1; i < n*n; i++){
          secondSum += a[j][i] * u[i];
        }
        //Bestimme neues U
        float newU = (h*h*functionF((j/n+1)*h,(j%n+1)*h) - firstSum - secondSum)/a[j][j];
        //Berechne Fehler
        float diff = newU-u[j];
        if( diff < 0)
            diff = -1* diff;
        if(fehler< diff);
          fehler = diff;
        //setze neuen u-Wert
        u[j] = newU;
    }
  }

	//TODO:Timerende




  for(int i = 0; i < n; i++){
    if(i ==0){
      for(int j = 0; j <n+2; j++){
  		    printf(" %f ",0.0f);
      }
      printf ("\n");
    }
    for(int j = n-1; j > -1; j--){
      if(j == n-1)printf(" %f ",0.0f);

      if(u[i +j*n] < 0)
		    printf("%f ",u[i +j*n]);
      else
		    printf(" %f ",u[i +j*n]);

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

  for(int i = 0; i < n*n; i++){
    free(a[i]);
  }
  free(a);
  free(u);
 return 0;
}



float functionF(float x, float y) { // x and y should be in (0,1)
  return (M*M+N*N)*(4*M_PI*M_PI*sin(2*M*M_PI*x)*sin(2*N*M_PI*y));
}
