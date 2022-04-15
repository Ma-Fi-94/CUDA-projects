#define N (100*1024+5)

#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include "cudaspecs.h"

// The "kernel" to run on the device
__global__ void dot (float *a, float *b, float *c) {
    __shared__ float tmp[MAX_THREADS_PER_BLOCK];
       
    // Element-wise parallel multiplication across threads
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < N) {
        tmp[threadIdx.x] = a[index] * b[index];
    } else {
        // In the last thread blocks, we might have some
        // empty threads at the ends.
        // Thus, we fill the corresponding vector elements
        // with zero, so that we may safely add up the
        // complete vector later.
        tmp[threadIdx.x] = 0;
    }
    
    // All thread wait here, until all threads reach this line
    __syncthreads();
    
    // Thread 0 sums up the products and writes sum back to *c
    if (threadIdx.x == 0) {
        float sum = 0;
        for (int i = 0; i < blockDim.x; i++) {
            sum += tmp[i];
        }
        // Add result of this threadblock to the overall sum
        // Needs to be atomic to avoid race conditions.
        atomicAdd(c, sum);
    }
}


int main() {
    // Memory size per vector
    int size = N * sizeof(float);
    
    // Allocate host memory
    float *a, *b, *c;
    a = (float*) malloc(size);
    b = (float*) malloc(size);
    c = (float*) malloc(sizeof(float));
    *c = 0;
    
    // Initialise randomly
    srand(time(NULL));
    for (int i = 0; i < N; i++){
        a[i] = 2; //rand() % 1000;
        b[i] = 3; //rand() % 1000;
    }
    
    // Allocate device memory
    // This needs a pointer to a pointer, hence we
    // pass the _address_ of our pointer
    float *dev_a, *dev_b, *dev_c; 
    cudaMalloc((void**) &dev_a, size);
    cudaMalloc((void**) &dev_b, size);
    cudaMalloc((void**) &dev_c, sizeof(float));
    
    // Copy to device
    cudaMemcpy(dev_a, a, size, cudaMemcpyHostToDevice);
    cudaMemcpy(dev_b, b, size, cudaMemcpyHostToDevice);
    cudaMemcpy(dev_c, c, size, cudaMemcpyHostToDevice);
    
    // Launch the "kernel"
    int nb_threads_per_block = MAX_THREADS_PER_BLOCK;
    int nb_threadblocks = 1+(N / MAX_THREADS_PER_BLOCK);
    printf("N=%i, thus %i blocks a %i threads.\n", N, nb_threadblocks, nb_threads_per_block);
    printf("Last thread only computes %i elements\n", N % MAX_THREADS_PER_BLOCK);
    assert (nb_threadblocks <= MAX_GRIDSIZE_1D);
    assert (nb_threads_per_block <= MAX_THREADS_PER_BLOCK);
    dot<<<nb_threadblocks, nb_threads_per_block>>>(dev_a, dev_b, dev_c);
    
    // Copy back
    cudaMemcpy(c, dev_c, sizeof(float), cudaMemcpyDeviceToHost);
    
    // The result
    printf("Result: %f, should be %i\n", *c, N*6);
    assert (*c == N*6);

    
    // Free device memory
    cudaFree(dev_a);
    cudaFree(dev_b);
    cudaFree(dev_c);

    // Free host memory
    free(a);
    free(b);
    free(c);

    // Done.
    return 0;
}
