
/*
 *	gcc -Wall -o pi pi.c -lm -fopenmp
*/

#include <omp.h>
#include <stdio.h>

static long num_steps = 1000000000;


void piWithAtomic() {

	long i;
	double x, pi, sum, step;
	
	sum = 0.0;
	step = 1.0 / (double) num_steps;
	double temp = 0.0;
	#pragma omp parallel
	{
		#pragma omp for
		for (i = 1; i <= num_steps; i++) 
		{
			x = (i-0.5) * step;
			temp = 4.0/(1.0+x*x);
			#pragma omp atomic update
			sum += temp;
			
		}
	}

	pi = step * sum;
	printf("Atomic-PI: %f\n", pi);
}
void piWithCritical() {

	long i;
	double x, pi, sum, step, temp;
	
	sum = 0.0;
	step = 1.0 / (double) num_steps;
	
	#pragma omp parallel
	{
		#pragma omp parallel for
		for (i = 1; i <= num_steps; i++) 
		{
			x = (i-0.5) * step;
			temp = 4.0/(1.0+x*x);
			#pragma omp critical 
			{
				sum += temp;	
			}
			
		}
	}

	pi = step * sum;
	printf("Critical-PI: %f\n", pi);
}
void piWithMaster() {

	long i;
	double x, pi, sum, step;
	
	sum = 0.0;
	step = 1.0 / (double) num_steps;
	double data[omp_get_num_threads()];

	#pragma omp parallel
	{
		#pragma omp for
		for (i = 1; i <= num_steps; i++) 
		{
			x = (i-0.5) * step;
			sum = sum + 4.0/(1.0+x*x);
		}
		printf("Sum: %f\n", sum);
		data[omp_get_thread_num()] = sum;
		#pragma omp master 
		{
			for(i = 0; i <= omp_get_thread_num(); i++) {
				sum += data[i];
			}
		}
	}

	pi = step * sum;

	printf("Master-PI: %f\n", pi);
}
void piWithReduction() {
	long i;
	double x, pi, sum, step;
	
	sum = 0.0;
	step = 1.0 / (double) num_steps;
	
	#pragma omp parallel
	{
		#pragma omp for reduction(+:sum) private(x)
		for (i = 1; i <= num_steps; i++) 
		{
			x = (i-0.5) * step;
			sum = sum + 4.0/(1.0+x*x);
		}
	}
	pi = step * sum;
	
	printf("ReductionPI: %f\n", pi);
}

void piNormal() {
	long i;
	double x, pi, sum, step;
	
	sum = 0.0;
	step = 1.0 / (double) num_steps;
	
	for (i = 1; i <= num_steps; i++) 
	{
		x = (i-0.5) * step;
		sum = sum + 4.0/(1.0+x*x);
	}
	pi = step * sum;
	
	printf("PI: %f\n", pi);
}


int main() {
	//piNormal();
	//piWithReduction();	// time 0m3,201s 4T
	//piWithMaster();
	//piWithCritical(); // time 7m2,181 4T
	//piWithAtomic(); // time 1m36,572s 4T
	return 0;
}
