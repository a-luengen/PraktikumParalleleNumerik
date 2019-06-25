
/*
 * Compilation:
 * 		w/ image output: gcc -Wall -o mandelbrot -D IMAGE_OUTPUT mandelbrot.c -lm
 * 			gcc -Wall -fopenmp -o mandelbrot -D IMAGE_OUTPUT mandelbrot.c -lm
 * 		w/o image output: gcc -Wall -o mandelbrot mandelbrot.c -lm
 */

#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#define N 10000 /* NxN grid size */
#define maxiter 1000


typedef struct {
	float re, im;
} complex;

float cabs1(complex x) {
	return sqrt(x.re * x.re + x.im * x.im);
}

complex add(complex x, complex y) {
	complex r;
	r.re = x.re + y.re; r.im = x.im + y.im;
	return r;
}

complex mult(complex x, complex y) {
	complex r;
	r.re = x.re * y.re - x.im * y.im;
	r.im = x.re * y.im + x.im * y.re;
	return r;
}

void normalMandelbrot() {
	complex z, kappa;
	int i, j, k;
	int *T;
	
	T = (int*) malloc(sizeof(int)*N*N);
	
	printf("Starting calculation for N=%d...\n", N);
	
	for (i=0; i<N; i++) {
		for (j=0; j<N; j++) {
			z.re = kappa.re = (4.0 * (i - N/2)) / N;
			z.im = kappa.im = (4.0 * (j - N/2)) / N;
			
			for (k=0; ; k++) {
				if (cabs1(z) > 2 || k == maxiter) {
					T[i * N + j] = (k * 256) / maxiter;
					break;
				}
				z = add( mult(z,z), kappa);
			}
		}
	}
	
	#ifdef IMAGE_OUTPUT
	printf("Writing simple image file...\n");
	
	FILE *f = fopen("output.ppm", "w");
	fprintf(f, "P3 %d %d 255 ", N, N);
	for (i=0; i<N; i++) {
		for (j=0; j<N; j++) {
			fprintf(f, "%d %d %d ", T[i*N+j]%256, T[i*N+j]%256, T[i*N+j]%256);
		}
	}
	fclose(f);
	#endif
	
	free(T);
}

void criticalMandelbrot() {
	complex z, kappa;
	int i, j, k;
	int *T;
	
	T = (int*) malloc(sizeof(int)*N*N);
	
	printf("Starting calculation for N=%d...\n", N);
	
	#pragma omp parallel
	{
		#pragma omp for private(z, kappa, k) collapse(2)
		for (i=0; i<N; i++) {
			for (j=0; j<N; j++) {
				z.re = kappa.re = (4.0 * (i - N/2)) / N;
				z.im = kappa.im = (4.0 * (j - N/2)) / N;

				for (k=0; ; k++) {
					if (cabs1(z) > 2 || k == maxiter) {
						T[i * N + j] = (k * 256) / maxiter;
						break;
					}
					z = add( mult(z,z), kappa);
				}
			}
		}
	}


	
	#ifdef IMAGE_OUTPUT
	printf("Writing simple image file...\n");
	
	FILE *f = fopen("output.ppm", "w");
	fprintf(f, "P3 %d %d 255 ", N, N);
	for (i=0; i<N; i++) {
		for (j=0; j<N; j++) {
			fprintf(f, "%d %d %d ", T[i*N+j]%256, T[i*N+j]%256, T[i*N+j]%256);
		}
	}
	fclose(f);
	#endif
	
	free(T);
}

int main() {

	//normalMandelbrot();			// 0m6,667s 
	criticalMandelbrot();			// 0m6,293s 1T, 0m4,621s 2T, 0m4,317s 4T, 0m2,927s 8T

	return 0;
}
