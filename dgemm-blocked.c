/* 
    Please include compiler name below (you may also include any other modules you would like to be loaded)

COMPILER= gnu

    Please include All compiler flags and libraries as you want them run. You can simply copy this over from the Makefile's first few lines
 
CC = cc
OPT = -O3
CFLAGS = -Wall -std=gnu99 $(OPT)
MKLROOT = /opt/intel/composer_xe_2013.1.117/mkl
LDLIBS = -lrt -Wl,--start-group $(MKLROOT)/lib/intel64/libmkl_intel_lp64.a $(MKLROOT)/lib/intel64/libmkl_sequential.a $(MKLROOT)/lib/intel64/libmkl_core.a -Wl,--end-group -lpthread -lm

*/

const char* dgemm_desc = "Simple blocked dgemm.";
#include <emmintrin.h>
#if !defined(BLOCK_SIZE)
#define BLOCK_SIZE 64
#endif

#define min(a,b) (((a)<(b))?(a):(b))

/* This auxiliary subroutine performs a smaller dgemm operation
 *  C := C + A * B
 * where C is M-by-N, A is M-by-K, and B is K-by-N. */

static void do_block (int lda, int M, int N, int K, double* A, double* B, double* C)
{
    /* For each row i of A */
    for (int i = 0; i < M; ++i)
        /* For each column j of B */
        for (int j = 0; j < N; ++j)
        {
            /* Compute C(i,j) */
            double cij = C[i+j*lda];
            for (int k = 0; k < K; ++k)
                cij += A[i+k*lda] * B[k+j*lda];
            C[i+j*lda] = cij;
        }
}


void do_block_fast (int lda, int M, int N, int K, double* A, double* B, double* C)
{
    static double a[BLOCK_SIZE*BLOCK_SIZE] __attribute__ ((aligned (16)));
    //static double b[BLOCK_SIZE*BLOCK_SIZE] __attribute__ ((aligned (128)));


    static double temp[2] __attribute__ ((aligned (16)));
    double tmpor = 0;



    __m128d vecA;
    __m128d vecB;
    __m128d vecC;

    __m128d vecAA;
    __m128d vecBB;
    __m128d vecCC;
    __m128d result;


//make a local aligned copy of A's block;
    for( int i = 0; i < M; i++ )
        for( int j = 0; j < K; j++ )
            a[j+i*BLOCK_SIZE] = A[i+j*lda];

    //for (int j = 0; j < K; j++)
      //  for (int i = 0; i < N; i++)
       //     b[i+j*BLOCK_SIZE] = B[i+j*lda];


/* For each row i of A */
    for (int i = 0; i < M; ++i) {

/* For each column j of B */
        for (int j = 0; j < N; ++j) {

/* Compute C(i,j) */
            double cij = C[i + j * lda];


            for (int k = 0; k < K; k +=4) {
                vecA = _mm_load_pd(&a[k + i * BLOCK_SIZE]);
                vecB = _mm_loadu_pd(&B[k + j * lda]);
                vecC = _mm_mul_pd(vecA, vecB);

                
                vecAA = _mm_load_pd(&a[k + 2 + i * BLOCK_SIZE]);


                vecBB = _mm_loadu_pd(&B[k + 2 + j * lda]);


                vecCC = _mm_mul_pd(vecAA, vecBB);


                result = _mm_add_pd(vecC, vecCC);

                _mm_store_pd(&temp[0], result);
                cij += temp[0];
                cij += temp[1];
                //cij += a[i+k*BLOCK_SIZE] * B[k+j*lda];
                //cij += a[i+(k+1)*BLOCK_SIZE] * B[(k+1)+j*lda];
            }


            C[i + j * lda] += cij;

        }

        }


}








/* This routine performs a dgemm operation
 *  C := C + A * B
 * where A, B, and C are lda-by-lda matrices stored in column-major format. 
 * On exit, A and B maintain their input values. */  
void square_dgemm (int lda, double* A, double* B, double* C)
{
  /* For each block-row of A */ 
  for (int i = 0; i < lda; i += BLOCK_SIZE)
    /* For each block-column of B */
    for (int j = 0; j < lda; j += BLOCK_SIZE)
      /* Accumulate block dgemms into block of C */
      for (int k = 0; k < lda; k += BLOCK_SIZE)
      {
	/* Correct block dimensions if block "goes off edge of" the matrix */
	int M = min (BLOCK_SIZE, lda-i);
	int N = min (BLOCK_SIZE, lda-j);
	int K = min (BLOCK_SIZE, lda-k);

          if((M % BLOCK_SIZE == 0) && (N % BLOCK_SIZE == 0) && (K % BLOCK_SIZE == 0))
          {
              do_block_fast(lda, M, N, K, A + i + k*lda, B + k + j*lda, C + i + j*lda);
          }else{
              do_block(lda, M, N, K, A + i + k*lda, B + k + j*lda, C + i + j*lda);
          }


	/* Perform individual block dgemm */
//	do_block(lda, M, N, K, A + i + k*lda, B + k + j*lda, C + i + j*lda);
      }
}
