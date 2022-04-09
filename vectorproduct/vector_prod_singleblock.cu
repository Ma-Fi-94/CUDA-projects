#define N 1000
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
    __shared__ int tmp[N];
    
    // Element-wise parallel multiplication across threads
    tmp[threadIdx.x] = a[threadIdx.x] * b[threadIdx.x];
    
    // All thread wait here, until all threads reach this line
    __syncthreads();
    
    // Thread 0 sums up the products and writes sum back to *c
    if (threadIdx.x == 0) {
        int sum = 0;
        for (int i = 0; i < N; i++) {
            sum += tmp[i];
        }
        *c = sum;
    }
}


int main() {
    // Memory size per vector
    int size = N * sizeof(int);
    
    // Allocate host memory
    int *a, *b, *c;
    a = (int*) malloc(size);
    b = (int*) malloc(size);
    c = (int*) malloc(sizeof(int));
    
    // Initialise randomly
    srand(time(NULL));
    for (int i = 0; i < N; i++){
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
    int nb_threadblocks = 1;
    int nb_threads_per_block = N;
    assert (nb_threadblocks < MAX_GRIDSIZE_1D);
    assert (nb_threads_per_block < MAX_THREADS_PER_BLOCK);
    dot<<<nb_threadblocks, nb_threads_per_block>>>(dev_a, dev_b, dev_c);
    
    // Copy back
    cudaMemcpy(c, dev_c, sizeof(int), cudaMemcpyDeviceToHost);

    // Check
    printf("%i should be %i\n", *c, N*6);
    
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
