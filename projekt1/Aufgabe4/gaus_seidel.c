#include <stdlib.h>
#include <stdio.h>
#include <math.h>

/**
 * gcc: gcc -Wall -g -fopenmp -o gaus_seidel gaus_seidel.c -lm -D PRINT
 * 
 */

#define L 3
#define MAX_K 15

float f(float x, float y);
float u(float x, float y);
int equals(float x, float y);
int a(int x, int y);
int T(int x, int y);

float h = 0.0;
int n = 0;

int main(void)
{

    h = 1.0 / pow(2.0, L);
    n = ((int)pow(2, L)) - 1;

    float epsilon = 0.00001;
    float *u = malloc(sizeof(float) * n * n);
    float *u_new = malloc(sizeof(float) * n * n);
    // wähle Startvektor u0 aus IR^n

    for (int i = 0; i < n * n; i++)
        u[i] = 0.0;

    float sum1 = 0.0;
    float sum2 = 0.0;
    int i, flag, k = 0;

    // iteration loop for gaus-seidel
    do
    {   
        // iteration loop for over the whole vector
        for (int j = 0; j < n * n; j++)
        {
            // first sum
            sum1 = 0.0;
            for (i = 1; i <= j - 1; i++)
                sum1 += a(j, i) * u[i];

            // second sum
            sum2 = 0.0;
            for (i = j + 1; i < n * n; i++)
                sum2 += a(j, i) * u[i];

            u_new[j] = (1.0 / a(j, j)) * (h * h * f((j / n) * h, (j % n) * h) - sum1 - sum2);
        }

        // Betrag der Differenz aus uk und uk+1 > epsilon um weiter zu iterrieren
        flag = 0;
        for (int j = 0; j < n * n && flag == 0; j++)
        {
            if (fabs(u[j] - u_new[j]) > epsilon)
            {
                // continue gaus seidel iteration
                flag = 1;
            }
        }

        #ifdef PRINT 
        printf("|%2d", k);
        for(int i = 0; i < n * n; i++) {
            printf("|%0.8f", u_new[i]);
        }
        printf("|\n");
        #endif

        // switch old with new vector
        float* temp = u;
        u = u_new;
        u_new = temp;

        k++;
    } while (flag == 1);
    printf("Anzahl Iterationen: %d\n", k);
    free(u);
    return 0;
}

int equals(float x, float y)
{
#ifdef DEBUG
    printf("Compare: %0.8f , %0.8f epsilon = %0,8f \n", x, y, h);
#endif

    if (fabsf(x - y) >= h)
    {
        return 0;
    }
    return 1;
}

float u(float x, float y)
{
    if (equals(x, 0.0) || equals(x, 1.0) || equals(y, 0.0) || equals(y, 1.0))
        return 0.0;
    return 16.0 * x * (1.0 - x) * y * (1.0 - y);
}

float f(float x, float y)
{
    return 32.0 * (x * (1.0 - x) + y * (1.0 - y));
}

int a(int x, int y)
{

    // 1. Check if in T Matrix:
    if ((x / n) == (y / n))
    {
        // get value from T Matrix
        return T(x % n, y % n);
    }

    // 2. Check if in -I Matrix
    if (abs((x / n) - (y / n)) == 1)
    {
        // Get value from -I Matrix
        if (x % n == y % n)
        {
            return -1;
        }
    }
    // None of the above, return 0
    return 0;
}

int T(int x, int y)
{
    if (x == y)
        return 4;
    if (abs(x - y) == 1)
        return -1;
    return 0;
}
