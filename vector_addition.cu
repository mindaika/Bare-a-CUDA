#include <stdio.h>
#include "time.h"
#include <stdlib.h>
#include <limits.h>

/* The old-fashioned CPU-only way to add two vectors */
void add_vectors_host(int *result, int *a, int *b, int n) {
     for (int i=0; i<n; i++)
          result[i] = a[i] + b[i];
}

/* The kernel that will execute on the GPU */
__global__ void add_vectors_kernel(int *result, int *a, int *b, int n) {
     int idx = blockDim.x * blockIdx.x + threadIdx.x;
     // If we have more threads than the magnitude of our vector, we need to
     // make sure that the excess threads don't try to save results into
     // unallocated memory.
     if (idx < n)
          result[idx] = a[idx] + b[idx];
}

/* This function encapsulates the process of creating and tearing down the
 * environment used to execute our vector addition kernel. The steps of the
 * process are:
 *   1. Allocate memory on the device to hold our vectors
 *   2. Copy the vectors to device memory
 *   3. Execute the kernel
 *   4. Retrieve the result vector from the device by copying it to the host
 *   5. Free memory on the device
 */
void add_vectors_dev(int *result, int *a, int *b, int n) {
     // Step 1: Allocate memory
     int *a_dev, *b_dev, *result_dev;

     // Since cudaMalloc does not return a pointer like C's traditional malloc
     // (it returns a success status instead), we provide as it's first argument
     // the address of our device pointer variable so that it can change the
     // value of our pointer to the correct device address.
     cudaMalloc((void **) &a_dev, sizeof(int) * n);
     cudaMalloc((void **) &b_dev, sizeof(int) * n);
     cudaMalloc((void **) &result_dev, sizeof(int) * n);

     // Step 2: Copy the input vectors to the device
     cudaMemcpy(a_dev, a, sizeof(int) * n, cudaMemcpyHostToDevice);
     cudaMemcpy(b_dev, b, sizeof(int) * n, cudaMemcpyHostToDevice);

     // Step 3: Invoke the kernel
     // We allocate enough blocks (each 512 threads long) in the grid to
     // accommodate all `n` elements in the vectors. The 512 long block size
     // is somewhat arbitrary, but with the constraint that we know the
     // hardware will support blocks of that size.
     dim3 dimGrid((n + 512 - 1) / 512, 1, 1);
     dim3 dimBlock(512, 1, 1);
     add_vectors_kernel<<<dimGrid, dimBlock>>>(result_dev, a_dev, b_dev, n);

     // Step 4: Retrieve the results
     cudaMemcpy(result, result_dev, sizeof(int) * n, cudaMemcpyDeviceToHost);

     // Step 5: Free device memory
     cudaFree(a_dev);
     cudaFree(b_dev);
     cudaFree(result_dev);
     }

     void print_vector(int *array, int n) {
     int i;
     for (i=0; i<n; i++)
          printf("%d ", array[i]);
     printf("\n");
}

int main(void) {
     const int CONST_VEC = INT_MAX / 5;
     // It looks like the crossover is just above this. At 2.6e8, GPU calculation time cuts in half (roughly).

     int* a = (int *)malloc(CONST_VEC * sizeof(int));
     int* b = (int *)malloc(CONST_VEC * sizeof(int));
     int* host_result = (int *)malloc(CONST_VEC * sizeof(int));
     int* device_result = (int *)malloc(CONST_VEC * sizeof(int));
 
     clock_t ar = clock();
     srand((unsigned)time(0)); 
     
     for(int i=0; i<CONST_VEC; i++){ 
          a[i] = (rand()%100)+1;
	     b[i] = (rand()%100)+1;  
     }
     ar = clock() - ar;
     printf ("It took me %ld clicks (%f seconds).\n",ar,((float)ar)/CLOCKS_PER_SEC);
	
     printf("The CPU's answer: ");
     clock_t t = clock();
     add_vectors_host(host_result, a, b, CONST_VEC);
     t = clock() - t;
     //print_vector(host_result, CONST_VEC);
     printf ("It took me %ld clicks (%f seconds).\n",t,((float)t)/CLOCKS_PER_SEC);
    

     printf("The GPU's answer: ");
     clock_t t2 = clock();
     add_vectors_dev(device_result, a, b, CONST_VEC);
     t2 = clock() - t2;
     //print_vector(device_result, CONST_VEC);
     printf ("It took me %ld clicks (%f seconds).\n",t2,((float)t2)/CLOCKS_PER_SEC);
	
     return 0;
}
