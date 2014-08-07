#include <stdio.h>
#include "time.h"
#include <stdlib.h>
#include <limits.h>

const int CONST_VEC = 1024;
__constant__ int constArrayA[CONST_VEC];
__constant__ int constArrayB[CONST_VEC];

void CPU_mult(int *result, int *a, int *b, int N) {
    int sum;
	for (int row=0; row<N; row++){
		for (int col=0; col<N; col++){
			sum=0;
			for (int n=0; n<N; n++){
				sum += a[row*N+n]*b[n*N+col];
			}
			result[row*N+col] = sum;
		}
	}
}

__global__ void matrix_kernel(int* result, int* A, int* B, int N)
{
	int row = blockIdx.y * blockDim.y + threadIdx.y;
	int col = blockIdx.x * blockDim.x + threadIdx.x;
	if (row < N && col < N) {
		int sum = 0;
		for (int n = 0; n < N; n++) {
			sum += A[row*N+n] * B[n*N + col];
		}
		result[row*N+col] = sum;
	}
}

void mult_vectors(int *result, int *a, int *b, int n) {
     int *A_DEV, *B_DEV, *RESULT_DEV;

    cudaMalloc((void **) &A_DEV, sizeof(int) * n * n);
    cudaMalloc((void **) &B_DEV, sizeof(int) * n * n);
    cudaMalloc((void **) &RESULT_DEV, sizeof(int) * n * n);

     cudaMemcpy(A_DEV, a, sizeof(int) * n * n, cudaMemcpyHostToDevice);
     cudaMemcpy(B_DEV, b, sizeof(int) * n * n, cudaMemcpyHostToDevice);

     dim3 dimGrid((n + 512 - 1) / 512, 1, 1);
     dim3 dimBlock(512, 1, 1);
     matrix_kernel<<<dimGrid, dimBlock>>>(RESULT_DEV, A_DEV, B_DEV, n);

     cudaMemcpy(result, RESULT_DEV, sizeof(int) * n * n, cudaMemcpyDeviceToHost);

	 /*
	for(int i=0; i<CONST_VEC; i++) {
		for(int j=0; j<CONST_VEC; j++) {
			printf("%i", a[i * CONST_VEC + j]);
			printf("  ");
		}
		printf("\n");
	}
	 */
	 
     cudaFree(A_DEV);
     cudaFree(B_DEV);
     cudaFree(RESULT_DEV);     
}

void mult_vectors_const(int *result, int *a, int *b, int n) {
	int *RESULT_DEV;

	cudaMalloc((void **) &RESULT_DEV, sizeof(int) * n * n);

	cudaMemcpyToSymbol(constArrayA, a, sizeof(int) * n * n, cudaMemcpyHostToDevice);
	cudaMemcpyToSymbol(constArrayB, b, sizeof(int) * n * n, cudaMemcpyHostToDevice);

	dim3 dimGrid((n + 512 - 1) / 512, 1, 1);
	dim3 dimBlock(512, 1, 1);
	matrix_kernel<<<dimGrid, dimBlock>>>(RESULT_DEV, constArrayA, constArrayB, n);

	cudaMemcpyToSymbol(result, RESULT_DEV, sizeof(int) * n * n, cudaMemcpyDeviceToHost);

	cudaFree(RESULT_DEV);
}

int main(void) {    

	int* a = (int *)malloc(CONST_VEC *  CONST_VEC * sizeof(int));
	int* b = (int *)malloc(CONST_VEC * CONST_VEC * sizeof(int));
	int* result = (int *)malloc(CONST_VEC * CONST_VEC * sizeof(int));

	clock_t ar = clock();
	srand((unsigned)time(0)); 

	for(int i=0; i<CONST_VEC; i++) {
		for(int j=0; j<CONST_VEC; j++) {
			a[i * CONST_VEC + j] = 1;//(rand()%10);
			b[i * CONST_VEC + j] = 1;//(rand()%10);
		}
	}
	
	ar = clock() - ar;
	printf ("It took me %ld clicks (%f seconds).\n",ar,((float)ar)/CLOCKS_PER_SEC);

	printf("Without Constant memory: ");
	clock_t t = clock();
	mult_vectors(result, a, b, CONST_VEC);
	t = clock() - t;
	printf ("It took me %ld clicks (%f seconds).\n",t,((float)t)/CLOCKS_PER_SEC);
	/*
	for(int i=0; i<CONST_VEC; i++) {
		for(int j=0; j<CONST_VEC; j++) {
			printf("%i", result[i * CONST_VEC + j]);
			printf("  ");
		}
		printf("\n");
	}
	*/
	printf("With Constant memory: ");
	clock_t t2 = clock();
	mult_vectors_const(result, a, b, CONST_VEC);
	t2 = clock() - t2;
	printf ("It took me %ld clicks (%f seconds).\n",t2,((float)t2)/CLOCKS_PER_SEC);
	/*
	for(int i=0; i<CONST_VEC; i++) {
		for(int j=0; j<CONST_VEC; j++) {
			printf("%i", result[i * CONST_VEC + j]);
			printf("  ");
		}
		printf("\n");
	}
	*/
	
	printf("On the CPU: ");
	clock_t t3 = clock();
	CPU_mult(result, a, b, CONST_VEC);
	t3 = clock() - t3;
	printf ("It took me %ld clicks (%f seconds).\n",t3,((float)t3)/CLOCKS_PER_SEC);

	/*
	for(int i=0; i<CONST_VEC; i++) {
		for(int j=0; j<CONST_VEC; j++) {
			printf("%i", result[i * CONST_VEC + j]);
			printf("  ");
		}
		printf("\n");
	}
	*/

	return 0;
}