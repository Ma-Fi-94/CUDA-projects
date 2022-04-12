// TODO: This is cheating because we manually define
// how we split the elements across the blocks.
// This should instead be detected automatically
// for an arbitrary input length.
#define T 10
#define B 10

#define MAX_THREADS_PER_BLOCK 1024
#define MAX_GRIDSIZE_1D 2147483647
#define MAX_GRIDSIZE_2D 65535
#define MAX_GRIDSIZE_3D 65535
#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

// The "kernel" to run on the device
__global__ void dot (int *a, int *b, int *c) {
    // Memory shared between all threads of a thread block
    // TODO: malloc dynamically blockDim.x ints
    __shared__ int tmp[T];
       
    // Element-wise parallel multiplication across threads
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    tmp[threadIdx.x] = a[index] * b[index];
    
    // All thread wait here, until all threads reach this line
    __syncthreads();
    
    // Thread 0 sums up the products and writes sum back to *c
    if (threadIdx.x == 0) {
        int sum = 0;
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
    int size = T * B * sizeof(int);
    
    // Allocate host memory
    int *a, *b, *c;
    a = (int*) malloc(size);
    b = (int*) malloc(size);
    c = (int*) malloc(sizeof(int));
    *c = 0;
    
    // Initialise randomly
    srand(time(NULL));
    for (int i = 0; i < T*B; i++){
        a[i] = 2; //rand() % 1000;
        b[i] = 3; //rand() % 1000;
    }
    
    // Allocate device memory
    // This needs a pointer to a pointer, hence we
    // pass the _address_ of our pointer
    int *dev_a, *dev_b, *dev_c; 
    cudaMalloc((void**) &dev_a, size);
    cudaMalloc((void**) &dev_b, size);
    cudaMalloc((void**) &dev_c, sizeof(int));
    
    // Copy to device
    cudaMemcpy(dev_a, a, size, cudaMemcpyHostToDevice);
    cudaMemcpy(dev_b, b, size, cudaMemcpyHostToDevice);
    
    // Launch the "kernel"
    int nb_threadblocks = B;
    int nb_threads_per_block = T;
    assert (nb_threadblocks <= MAX_GRIDSIZE_1D);
    assert (nb_threads_per_block <= MAX_THREADS_PER_BLOCK);
    dot<<<nb_threadblocks, nb_threads_per_block>>>(dev_a, dev_b, dev_c);
    
    // Copy back
    cudaMemcpy(c, dev_c, sizeof(int), cudaMemcpyDeviceToHost);
    
    // The result
    printf("Result: %i, should be %i\n", *c, T*B*6);

    
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
