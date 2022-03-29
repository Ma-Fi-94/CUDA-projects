// From deviceQuery
// TODO: make header file for this
#define MAX_THREADS_PER_BLOCK 1024
#define MAX_GRIDSIZE_1D 2147483647
#define MAX_GRIDSIZE_2D 65535
#define MAX_GRIDSIZE_3D 65535

// Simulation parameters
#define XSIZE 750
#define YSIZE 750
#define MAXSTEP 100

#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <unistd.h>

// The "kernel" to run on the device
__global__ void propagate(int *lattice, int* lattice_new) {
    int X = threadIdx.x;
    int Y = blockIdx.x;
    int i = YSIZE * X + Y;
    lattice_new[i] = lattice[i];

    if (X > 0 & Y > 0 & X < XSIZE-1 & Y < YSIZE-1) {
        int n = lattice[i-YSIZE-1] + lattice[i-YSIZE] + lattice[i-YSIZE+1] +
                lattice[i-1] + lattice[i+1] + 
                lattice[i+YSIZE-1] + lattice[i+YSIZE] + lattice[i+YSIZE+1];
        
        if (lattice[i] == 0 & n == 3) {
            lattice_new[i] = 1;
        } else if (lattice[i] == 1 & n < 2) {
            lattice_new[i] = 0;
        } else if (lattice[i] == 1 & n > 3) {
            lattice_new[i] = 0;
        }        
    }
}


// Another "kernel" to run on the device
__global__ void update(int *lattice, int* lattice_new) {
    int X = threadIdx.x;
    int Y = blockIdx.x;
    int i = YSIZE * X + Y;
    lattice[i] = lattice_new[i];
}


int main() {
    // Allocate host memory
    int *lattice;
    lattice = (int*) calloc(XSIZE * YSIZE, sizeof(int));
      
    // Initialise host array
    srand(time(NULL));
    for (int i = 1; i < XSIZE-1; i++) {
        for (int j = 1; j < YSIZE-1; j++) {
            lattice[i*YSIZE+j] = rand() % 100 < 50;
        }
    }
    
    // TODO: Start timer
    
    // Allocate device memory
    // This needs a pointer to a pointer, hence we
    // pass the _address_ of our pointer
    int *d_lattice, *d_lattice_new; 
    cudaMalloc((void**) &d_lattice, sizeof(int) * XSIZE * YSIZE);
    cudaMalloc((void**) &d_lattice_new, sizeof(int) * XSIZE * YSIZE);
    
    
    // Copy lattice from host to device
    cudaMemcpy(d_lattice, lattice, sizeof(int) * XSIZE * YSIZE, cudaMemcpyHostToDevice);

    // Preparation and sanity checks for kernel launches
    int T = XSIZE;  // nb. threads per thread block
    int G = YSIZE;  // nb. thread blocks
    assert (T <= MAX_THREADS_PER_BLOCK);
    assert (G <= MAX_GRIDSIZE_1D);
    
    // Main simulation loop
    for (int i = 0; i <= MAXSTEP; i++) {               
        propagate<<<G,T>>>(d_lattice, d_lattice_new);
        update<<<G,T>>>(d_lattice, d_lattice_new);
    }
      
   
   // Deallocate device memory
   cudaFree(d_lattice);
   
   // TODO: Stop timer
   
   // Deallocate host memory
   free(lattice);
}

