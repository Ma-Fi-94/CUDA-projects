// From deviceQuery
// TODO: make header file for this
#define MAX_THREADS_PER_BLOCK 1024
#define MAX_GRIDSIZE_1D 2147483647
#define MAX_GRIDSIZE_2D 65535
#define MAX_GRIDSIZE_3D 65535

// Simulation parameters
#define XSIZE 35
#define YSIZE 120
#define MAXSTEP 10

#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>


// The "kernel" to run on the device
__global__ void propagate(int *lattice) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Handling arbitrary vector size
    if (tid < XSIZE * YSIZE){
        // TODO: magic
    }
}

void print_lattice(int* x) {
    for (int i = 0; i < XSIZE; i++) {
        for (int j = 0; j < YSIZE; j++) {
            printf("%s", x[i*XSIZE+j] ? "•" : " ");
        }
        printf("\n");
    }
}

int main() {
    // Allocate host memory
    int *lattice;
    lattice = (int*) malloc(sizeof(int) * XSIZE * YSIZE);
      
    
    // Initialise host array
    srand(time(NULL));
    for (int i = 0; i < XSIZE; i++) {
        for (int j = 0; j < YSIZE; j++) {
            lattice[i*XSIZE+j] = rand() % 100 < 10;
        }
    }
    
    
    // Allocate device memory
    // This needs a pointer to a pointer, hence we
    // pass the _address_ of our pointer
    int *d_lattice; 
    cudaMalloc((void**) &d_lattice, sizeof(int) * XSIZE * YSIZE);
    
    
    // Main simulation loop
    printf("\033[2J");
    for (int i = 0; i < MAXSTEP; i++) {
        // Print current state
        printf("\033[1;1HStep: %i / %i\n\n", i, MAXSTEP);
        print_lattice(lattice);
        getchar();    
        
        // Copy lattice from host to device
        cudaMemcpy(d_lattice, lattice, sizeof(int) * XSIZE * YSIZE, cudaMemcpyHostToDevice);

        
        // Kernel launch
        int T = XSIZE;  // nb. threads per thread block
        int G = YSIZE;  // nb. thread blocks
        assert (T <= MAX_THREADS_PER_BLOCK);
        assert (G <= MAX_GRIDSIZE_1D);
        propagate<<<G,T>>>(d_lattice);

        
        // Copy back
        cudaMemcpy(lattice, d_lattice, sizeof(int) * XSIZE * YSIZE, cudaMemcpyDeviceToHost);
    }
      
   
   // Deallocate device memory
   cudaFree(d_lattice);
   
   // Deallocate host memory
   free(lattice);
}

