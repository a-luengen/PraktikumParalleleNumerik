#include <stdio.h>

#include <stdlib.h>

#define FEHLERSCHRANKE 0.000001
//Exponent der Verfeinerung

float functionF(float x, float y);
float **allocateSquareMatrix(int size, int initialize, int n);
float *allocateVector(int size, int initialize);
void printSquareMatrix(float **matrix, int dim);
void printVector(float *vector, int length);
void printVectorInBlock(float *vector, int length, int blockLength);
void freeSquareMatrix(float **matrix, int dim);

/**
*      
*   n: Dimension of u matrix
*   fehlerSchranke: Abbruchbedingung für Gaus Seidel Verfahren
*   h: Verfeinerung/Gitterschrittweite
*   a: pointer to 2D-Array of Matrix a (currently not used, due 
*       to extraction of calculation pattern into algorithm)
*   u: pointer to u matrix
*/
void gaussSeidel(int n, float fehlerSchranke, float h, float **a, float *u)
{
    //TODO: Timeranfang
    float fehler = fehlerSchranke + 1;
    float diff = 0.0;
    float newU = 0.0;
    int n_emb = n + 2;
    float tempSum = 0.0;

    // embedd vector u for corner case
    float **u_emb = allocateSquareMatrix(n_emb * n_emb, 0, n_emb);

    // embedd vector u
    for (int i = 1; i < n_emb - 1; i++)
    {
        for (int j = 1; j < n_emb - 1; j++)
        {
            // copy value from u
            u_emb[i][j] = u[i - 1 + n * j - 1];
        }
    }
#ifdef PRINT
    // print embedded vector u
    printSquareMatrix(u_emb, n_emb);
#endif
    int count = 0;
    while (fehlerSchranke < fehler)
    {
        fehler = 0.0;
        int i_emb, j_emb;

        // using jacoby calculation for gaussSeidel with red-black chess structure
        // iterating over dimensions of u but use n_emb and increment i/j to access values in embedded vector
        // "black" colored elements first (start_iter = 0) and then the "red" colored elements (start_iter = 1)
        for (int start_iter = 0; start_iter < 2; start_iter++)
        {
#pragma omp parallel for private(tempSum, newU, diff, i_emb, j_emb) shared(fehler)
            for (int j = start_iter; j < n * n; j += 2)
            {

                // transform j into coordinates i_emb, j_emb where i_emb = column and j_emb = row in embedded vector/matrix
                // i_emb = j%n + 1
                // j_emb = j/n + 1
                i_emb = j % n + 1;
                j_emb = j / n + 1;

                // top element
                tempSum = u_emb[i_emb][j_emb - 1];
                // left element
                tempSum += u_emb[i_emb - 1][j_emb];
                // right element
                tempSum += u_emb[i_emb + 1][j_emb];
                // bottom element
                tempSum += u_emb[i_emb][j_emb + 1];

                // calc new value for u
                newU = (h * h * functionF((j / n + 1) * h, (j % n + 1) * h) + tempSum) / 4.0;

                // Calculate error
                diff = newU - u_emb[i_emb][j_emb];
                if (diff < 0) {
                    diff = -1 * diff;
                }
                #pragma omp critical
                {
                    // update this atomically
                    if (fehler < diff) {
                        fehler = diff;
                    }
                }

                //set new value for u in embedded vector
                u_emb[i_emb][j_emb] = newU;
            }
        }
        count++;
    }
    printf("Took %d -Iterations. \n", count);

#ifdef PRINT
    // print embedded vector u
    printSquareMatrix(u_emb, n_emb);
#endif

    // get values out of embedded vector
    for (int i = 0; i < n; i++)
    {
        for (int j = 0; j < n; j++)
        {
            u[i + j * n] = u_emb[i + 1][j + 1];
        }
    }
    freeSquareMatrix(u_emb, n_emb);
    free(u_emb);
}

int main()
{
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

    //Matrix A
    float **a = allocateSquareMatrix((n * n), 1, n);

#ifdef PRINT
    //printSquareMatrix(a, (n * n)*(n * n));
    printVector(u, (n * n));
#endif

    // executing gauss seidel verfahren
    gaussSeidel(n, FEHLERSCHRANKE, h, a, u);

#ifdef PRINT
    printSquareMatrix(a, (n * n));
#endif

    printVectorInBlock(u, (n * n), n);
    printVector(u, (n * n));

    freeSquareMatrix(a, (n * n));
    free(a);
    free(u);
    return 0;
}

float functionF(float x, float y)
{ // x and y should be in (0,1)
    return 32.0f * (x * (1.0f - x) + y * (1.0f - y));
}

float **allocateSquareMatrix(int size, int initialize, int n)
{
    float **tmp = (float **)malloc(size * sizeof(float *));

    for (int i = 0; i < size; i++)
    {
        tmp[i] = (float *)malloc(size * sizeof(float));
    }

    if (initialize)
    {
        for (int i = 0; i < size; i++)
        {
            for (int j = 0; j < size; j++)
            {
                tmp[i][j] = 0.0;
                if (i == j)
                    tmp[i][j] = 4.0;

                if (i + n == j || i == j + n || i + 1 == j || i == j + 1)
                    tmp[i][j] = -1.0;

                if ((i % n == 0 && j == i - 1) || (i == j - 1 && j % n == 0))
                    tmp[i][j] = 0.0;
            }
        }
    }
    else
    {
        for (int i = 0; i < size; i++)
        {
            for (int j = 0; j < size; j++)
            {
                tmp[i][j] = 0.0;
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

float *allocateVector(int size, int initialize)
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
            printf("|%f", matrix[i][j]);
        }
        printf("|\n");
    }
}

void printVector(float *vector, int length)
{
    printf("Printing Vector with length = %d\n", length);
    for (int i = 0; i < length; i++)
        printf("|%f", vector[i]);

    printf("|\n");
}

void printVectorInBlock(float *vector, int length, int blockLength)
{
    printf("Printing Vector with length = %d\n", length);
    for (int i = 0; i < length / blockLength; i++)
    {
        for (int j = 0; j < blockLength; j++)
        {
            printf("|%f", vector[i + blockLength * j]);
        }
        printf("|\n");
    }
}