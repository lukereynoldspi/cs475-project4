#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>
#include <ctime>
#include <sys/time.h>
#include <sys/resource.h>
#include <omp.h>

// SSE stands for Streaming SIMD Extensions

#define SSE_WIDTH	4
#define ALIGNED		__attribute__((aligned(16)))


#define NUMTRIES	100

#ifndef ARRAYSIZE
#define ARRAYSIZE	1024*1024
#endif

ALIGNED float A[ARRAYSIZE];
ALIGNED float B[ARRAYSIZE];
ALIGNED float C[ARRAYSIZE];


void	SimdMul(    float *, float *,  float *, int );
void	NonSimdMul( float *, float *,  float *, int );
float	SimdMulSum(    float *, float *, int );
float	NonSimdMulSum( float *, float *, int );


int
main( int argc, char *argv[ ] )
{
	for( int i = 0; i < ARRAYSIZE; i++ )
	{
		A[i] = sqrtf( (float)(i+1) );
		B[i] = sqrtf( (float)(i+1) );
	}

	fprintf( stderr, "%12d\t", ARRAYSIZE );

	double maxPerformance = 0.;
	for( int t = 0; t < NUMTRIES; t++ )
	{
		double time0 = omp_get_wtime( );
		NonSimdMul( A, B, C, ARRAYSIZE );
		double time1 = omp_get_wtime( );
		double perf = (double)ARRAYSIZE / (time1 - time0);
		if( perf > maxPerformance )
			maxPerformance = perf;
	}
	double megaMults = maxPerformance / 1000000.;
	fprintf( stderr, "N %10.2lf\t", megaMults );
	double mmn = megaMults;


	maxPerformance = 0.;
	for( int t = 0; t < NUMTRIES; t++ )
	{
		double time0 = omp_get_wtime( );
		SimdMul( A, B, C, ARRAYSIZE );
		double time1 = omp_get_wtime( );
		double perf = (double)ARRAYSIZE / (time1 - time0);
		if( perf > maxPerformance )
			maxPerformance = perf;
	}
	megaMults = maxPerformance / 1000000.;
	fprintf( stderr, "S %10.2lf\t", megaMults );
	double mms = megaMults;
	double speedup = mms/mmn;
	fprintf( stderr, "(%6.2lf)\t", speedup );


	maxPerformance = 0.;
	float sumn, sums;
	for( int t = 0; t < NUMTRIES; t++ )
	{
		double time0 = omp_get_wtime( );
		sumn = NonSimdMulSum( A, B, ARRAYSIZE );
		double time1 = omp_get_wtime( );
		double perf = (double)ARRAYSIZE / (time1 - time0);
		if( perf > maxPerformance )
			maxPerformance = perf;
	}
	double megaMultAdds = maxPerformance / 1000000.;
	fprintf( stderr, "N %10.2lf\t", megaMultAdds );
	mmn = megaMultAdds;


	maxPerformance = 0.;
	for( int t = 0; t < NUMTRIES; t++ )
	{
		double time0 = omp_get_wtime( );
		sums = SimdMulSum( A, B, ARRAYSIZE );
		double time1 = omp_get_wtime( );
		double perf = (double)ARRAYSIZE / (time1 - time0);
		if( perf > maxPerformance )
			maxPerformance = perf;
	}
	megaMultAdds = maxPerformance / 1000000.;
	fprintf( stderr, "S %10.2lf\t", megaMultAdds );
	mms = megaMultAdds;
	speedup = mms/mmn;
	fprintf( stderr, "(%6.2lf)\n", speedup );
	//fprintf( stderr, "[ %8.1f , %8.1f , %8.1f ]\n", C[ARRAYSIZE-1], sumn, sums );

	return 0;
}


void
NonSimdMul( float *A, float *B, float *C, int n )
{
	for( int i = 0; i < n; i++ )
	{
		C[i] = A[i] * B[i];
	}
}

float
NonSimdMulSum( float *A, float *B, int n )
{
	float sum = 0.0;

    for( int i = 0; i < n; i++ )
    {
        sum += A[i] * B[i];
    }

    return sum;
}


void
SimdMul( float *a, float *b,   float *c,   int len )
{
	int limit = ( len/SSE_WIDTH ) * SSE_WIDTH;
	__asm
	(
		".att_syntax\n\t"
		"movq    -24(%rbp), %r8\n\t"		// a
		"movq    -32(%rbp), %rcx\n\t"		// b
		"movq    -40(%rbp), %rdx\n\t"		// c
	);

	for( int i = 0; i < limit; i += SSE_WIDTH )
	{
		__asm
		(
			".att_syntax\n\t"
			"movups	(%r8), %xmm0\n\t"	// load the first sse register
			"movups	(%rcx), %xmm1\n\t"	// load the second sse register
			"mulps	%xmm1, %xmm0\n\t"	// do the multiply
			"movups	%xmm0, (%rdx)\n\t"	// store the result
			"addq $16, %r8\n\t"
			"addq $16, %rcx\n\t"
			"addq $16, %rdx\n\t"
		);
	}

	for( int i = limit; i < len; i++ )
	{
		c[i] = a[i] * b[i];
	}
}



float SimdMulSum(float *a, float *b, int len)
{
    float sum[4] = {0.0, 0.0, 0.0, 0.0};
    int limit = (len / SSE_WIDTH) * SSE_WIDTH;

    __asm__ volatile
    (
        "movups (%0), %%xmm2\n" // load xmm2 with initial sum values
        :
        : "r" (sum) // input: sum array
        : "%xmm2" // xmm2 is clobbered
    );

    for (int i = 0; i < limit; i += SSE_WIDTH)
    {
        __asm__ volatile
        (
            "movups (%0), %%xmm0\n" // load the first sse register
            "movups (%1), %%xmm1\n" // load the second sse register
            "mulps %%xmm1, %%xmm0\n" // do the multiply
            "addps %%xmm0, %%xmm2\n" // do the add
            :
            : "r" (&a[i]), "r" (&b[i]) // input: pointers to a[i] and b[i]
            : "%xmm0", "%xmm1", "%xmm2" // xmm0, xmm1, xmm2 are clobbered
        );
    }

    __asm__ volatile
    (
        "movups %%xmm2, (%0)\n" // copy the sums back to sum[]
        :
        : "r" (sum) // output: sum array
        : "%xmm2" // xmm2 is clobbered
    );

    float totalSum = sum[0] + sum[1] + sum[2] + sum[3];
    for (int i = limit; i < len; i++)
    {
        totalSum += a[i] * b[i];
    }

    return totalSum;
}

