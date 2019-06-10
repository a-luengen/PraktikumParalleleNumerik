
/*
 * compile: gcc -Wall -g -fopenmp -o demo_with_bugs demo_with_bugs.c -lm
 * with sanitize: gcc -Wall -g -fopenmp -o demo_with_bugs demo_with_bugs.c -lm -fsanitize=thread
 */

#include <stdio.h>
#include <math.h> 
#include <omp.h>

#define N 100

float work(int a) {
  return a+1;
}

int main(void) {
	float a[N], b[N], x, f, sum, sum_expected;
	int i;
	
	/* Example 1 (this example contains an OpenMP parallelization error) */
	/* --------- */
	
	
	/* PROBLEM:
		Race-Condition auf Array a mit index i, bei ungünstiger Laufzeit kann es dazu kommen,
		das a[i] durch einen Thread geschrieben wird und von einem anderen Thread aus versucht wird, darauf 
		auf diesen Wert in der nachfolgenden Anweisung zuzugreifen.

		LÖSUNG:
		Aufteilen in zwei parallele Schleifen

	*/
	a[0] = 0;
	# pragma omp parallel for
	for (i=1; i<N; i++) {	
		a[i] = 2.0*i*(i-1);
		//b[i] = a[i] - a[i-1];
	}
	# pragma omp parallel for
	for(i=1; i<N; i++) {
		b[i] = a[i] - a[i-1];
	}
	
	/* testing the correctness of the numerical result: */
	sum=0; for (i=1; i<N; i++) { sum = sum + b[i]; }
	sum_expected = a[N-1]-a[0];
	printf("Exa.1: sum  computed=%8.1f,  expected=%8.1f,  difference=%8.5f \n", 
							sum,  sum_expected, sum-sum_expected); 
	
	
	/* Example 2 (this example contains an OpenMP parallelization error) */
	/* --------- */
	
	/**
	 * PROBLEM:
	 * Threads existieren in der gesamten parallelen Region.
	 * Das nowait statement der ersten Parallelisierten For-Loop bewirkt, 
	 * das Threads nach dem sie fertig mit ihren Teilaufgaben sind, direkt
	 * die Bearbeitung der nächsten For-Loop beginnen und es wieder zu 
	 * Race-Conditions auf Array a mit index i kommen kann.
	 * LÖSUNG:
	 * nowait entfernen und jede Schleife mit impliziter Barriere
	 */

	a[0] = 0;
	# pragma omp parallel
	{
		#   pragma omp for //nowait
		for (i=1; i<N; i++) {
			a[i] = 3.0*i*(i+1);
		}
		
		#   pragma omp for
		for (i=1; i<N; i++) {
			b[i] = a[i] - a[i-1];
		}
	}

	/* testing the correctness of the numerical result: */
	sum=0; for (i=1; i<N; i++) { sum = sum + b[i]; }
	sum_expected = a[N-1]-a[0];
	printf("Exa.2: sum  computed=%8.1f,  expected=%8.1f,  difference=%8.5f \n", 
							sum,  sum_expected, sum-sum_expected); 
	
	
	/* Example 3 (this example contains an OpenMP parallelization error) */
	/* --------- */
	
	/*
		PROBLEM:
		Variable x ist global definiert somit implizit shared zwischen den Threads.
		Bei ungünstiger Ausführungsreihenfolge besteht Race-Condition auf Variable x,
		da sowohl lesen als auch schreiben auf x zugegriffen wird.
		LÖSUNG:
		x als private deklarieren, damit jeder Thread seine eigene Kopie hat
	
	*/
	# pragma omp parallel for private(x)
	for (i=1; i<N; i++) {
		x = sqrt(b[i]) - 1;
		a[i] = x*x + 2*x + 1;
	}
	
	/* testing the correctness of the numerical result: */
	sum=0; for (i=1; i<N; i++) { sum = sum + a[i]; }
	/* sum_expected = same as in Exa.2 */
	printf("Exa.3: sum  computed=%8.1f,  expected=%8.1f,  difference=%8.5f \n", 
							sum,  sum_expected, sum-sum_expected); 
	
	/* Example 4 (this example contains an OpenMP parallelization error) */
	/* --------- */
	
	/*
		PROBLEM:
		f ist global definiert und wird für jeden Thread durch private als eigene Kopie
		deklariert. Allerdings wird durch die verwendung von private lediglich eine
		uninitialisierte kopie der variable f für jeden Thread angelegt.
		Dadurch hat nur der main-thread den Wert 2 in der Kopie von f stehen.
		Zudem wird der Wert von x nicht aus der parallelen Region wieder rausgeschrieben, 
		sondern gelöscht. Wodurch der zuletzt geschriebene Wert von x außerhalb nicht sichtbar ist.
		
		LÖSUNG:
		firstprivate statement verwenden um den korrekten Wert in die Kopieen zu initialisieren.
		lastprivate statement verwenden um den korrekten Wert aus der parallelen Region für
		x in den Main-Thread zu übernehmen.
	*/
	f = 2;
	# pragma omp parallel for firstprivate(f) lastprivate(x)
	for (i=1; i<N; i++) {
		x = f * b[i];
		a[i] = x - 7;
	}
	a[0] = x;
	
	/* testing the correctness of the numerical result: */
	printf("Exa.4: a[0] computed=%8.1f,  expected=%8.1f,  difference=%8.5f \n", 
							a[0],        2*b[N-1],   a[0] - 2*b[N-1] );
	
	
	/* Example 5 (this example contains an OpenMP parallelization error) */


	/*
		PROBLEM: Race-Condition auf sum

		LÖSUNG: sum
	
	*/
	/* --------- */
	
	sum = 0;
	# pragma omp parallel for
	for (i=1; i<N; i++) {
		sum = sum + b[i];
	}
	
	/* testing the correctness of the numerical result: */
	/* sum_expected = same as in Exa.2 */
	printf("Exa.5: sum  computed=%8.1f,  expected=%8.1f,  difference=%8.5f \n", 
							sum,  sum_expected, sum-sum_expected); 
	
	return 0;
}

