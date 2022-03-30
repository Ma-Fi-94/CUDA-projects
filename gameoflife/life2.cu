// From deviceQuery
// TODO: make header file for this
#define MAX_THREADS_PER_BLOCK 1024
#define MAX_GRIDSIZE_1D 2147483647
#define MAX_GRIDSIZE_2D 65535
#define MAX_GRIDSIZE_3D 65535

// Simulation parameters
#define XSIZE 40
#define YSIZE 160
#define MAXSTEP 1000
#define DELAY_MS 100

#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <unistd.h>

// The "kernel" to run on the device
__global__ void propagate(int *lattice, int* lattice_new) {
    int i = blockIdx.x;
    lattice_new[i] = lattice[i];
    
    int X = i / YSIZE;
    int Y = i % YSIZE;

    if (X > 0 & Y > 0 & X < XSIZE-1 & Y < YSIZE-1) {
        int n = lattice[i-YSIZE-1] + lattice[i-YSIZE] + lattice[i-YSIZE+1] +
                lattice[i-1] + lattice[i+1] + 
                lattice[i+YSIZE-1] + lattice[i+YSIZE] + lattice[i+YSIZE+1];
        
        if (lattice[i] == 0 & n == 3) {
            lattice_new[i] = 1;
            return;
        }
        
        if (lattice[i] == 1 & n < 2) {
            lattice_new[i] = 0;
            return;
        }

        if (lattice[i] == 1 & n > 3) {
            lattice_new[i] = 0;
            return;
        }
        return;
    }
    
    return;


}


void print_lattice(int* x) {
    printf("+");
    for (int j = 0; j < YSIZE; j++) {
        printf("-");
    }
    printf("+\n");

    for (int i = 0; i < XSIZE; i++) {
        printf("|");
        for (int j = 0; j < YSIZE; j++) {
            printf("%s", x[i*YSIZE+j] ? "■" : " ");
        }
        printf("|\n");
    }
    
    printf("+");
    for (int j = 0; j < YSIZE; j++) {
        printf("-");
    }
    printf("+\n");


}


int main() {
    // Allocate host memory
    int *lattice;
    lattice = (int*) calloc(XSIZE * YSIZE, sizeof(int));
      
    // Initialise host array
    srand(time(NULL));
    for (int i = 1; i < XSIZE-1; i++) {
        for (int j = 1; j < YSIZE-1; j++) {
            lattice[i*YSIZE+j] = rand() % 100 < 33;
        }
    }
    
    // Allocate device memory
    // This needs a pointer to a pointer, hence we
    // pass the _address_ of our pointer
    int *d_lattice, *d_lattice_new; 
    cudaMalloc((void**) &d_lattice, sizeof(int) * XSIZE * YSIZE);
    cudaMalloc((void**) &d_lattice_new, sizeof(int) * XSIZE * YSIZE);
    
    
    // Main simulation loop
    printf("\033[2J");
    for (int i = 0; i <= MAXSTEP; i++) {
        // Print current state
        printf("\033[1;1HStep: %i / %i\n\n", i, MAXSTEP);
        print_lattice(lattice);
        //usleep(1000*DELAY_MS);
        
        // Copy lattice from host to device
        cudaMemcpy(d_lattice, lattice, sizeof(int) * XSIZE * YSIZE, cudaMemcpyHostToDevice);

        
        // Kernel launch
        int T = 1;  // nb. threads per thread block
        int G = XSIZE*YSIZE;  // nb. thread blocks
        assert (G <= MAX_GRIDSIZE_1D);
        propagate<<<G,T>>>(d_lattice, d_lattice_new);

        
        // Copy back
        cudaMemcpy(lattice, d_lattice_new, sizeof(int) * XSIZE * YSIZE, cudaMemcpyDeviceToHost);
    }
      
   
   // Deallocate device memory
   cudaFree(d_lattice);
   
   // Deallocate host memory
   free(lattice);
}

