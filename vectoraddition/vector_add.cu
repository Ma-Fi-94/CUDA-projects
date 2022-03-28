#define N 10000000
#define MAX_ERR 1e-6
#define MAX_THREADS_PER_BLOCK 1024
#define MAX_GRIDSIZE_1D 2147483647
#define MAX_GRIDSIZE_2D 65535
#define MAX_GRIDSIZE_3D 65535
#include <assert.h>
#include <stdio.h>

// The "kernel" to run on the device
__global__ void vector_add(float *out, float *a, float *b, int n) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Handling arbitrary vector size
    if (tid < n){
        out[tid] = a[tid] + b[tid];
    }
}


int main() {
    // Allocate host memory
    float *a, *b, *out;
    a   = (float*) malloc(sizeof(float) * N);
    b   = (float*) malloc(sizeof(float) * N);
    out = (float*) malloc(sizeof(float) * N);
      
    
    // Initialise host array
    for (int i = 0; i < N; i++){
        a[i] = 1.0f;
        b[i] = 2.0f;
    }
   
   
    // Allocate device memory
    // This needs a pointer to a pointer, hence we
    // pass the _address_ of our pointer
    float *d_a, *d_b, *d_out; 
    cudaMalloc((void**) &d_a, sizeof(float) * N);
    cudaMalloc((void**) &d_b, sizeof(float) * N);
    cudaMalloc((void**) &d_out, sizeof(float) * N);
    
    
    // Transfer data from host to device memory
    cudaMemcpy(d_a, a, sizeof(float) * N, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b, sizeof(float) * N, cudaMemcpyHostToDevice);    
    
    
    // Executing kernel 
    int T = 1024;  // nb. threads per thread block
    int M = N/T + 1;  // nb. thread blocks
    assert (T <= MAX_THREADS_PER_BLOCK);
    assert (M <= MAX_GRIDSIZE_1D);
    vector_add<<<M,T>>>(d_out, d_a, d_b, N);
    
    
    // Transfer data from device memory back to host memory
    cudaMemcpy(out, d_out, sizeof(float) * N, cudaMemcpyDeviceToHost);
    

    // Deallocate device memory
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_out);
    
    
    // Verification
    for(int i = 0; i < N; i++){
        assert(fabs(out[i] - a[i] - b[i]) < MAX_ERR);
    }
    printf("out[0] = %f\n", out[0]);
    printf("PASSED\n");

    
    // Deallocate host memory
    free(a); 
    free(b); 
    free(out);


}

