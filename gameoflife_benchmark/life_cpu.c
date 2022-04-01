// Simulation parameters
#define MAXSTEP 100

#include <time.h>
#include <unistd.h>
#include <stdlib.h>
#include <stdio.h>
#include <assert.h>


void propagate_and_update(int *lattice, int* lattice_new, int XSIZE, int YSIZE) {
    for (int x = 0; x < XSIZE; x++) {
        for (int y = 0; y < YSIZE; y++) { 
            int i = YSIZE * x + y;
            lattice_new[i] = lattice[i];
            
            if (x > 0 & y > 0 & x < XSIZE-1 & y < YSIZE-1) {
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
    }   
    
    for (int i = 0; i < XSIZE*YSIZE; i++) {
        lattice[i] = lattice_new[i];
    }
}



int main(int argc, char *argv[]) {
    assert(argc==3);   
    int XSIZE = atoi(argv[1]);
    int YSIZE = atoi(argv[2]);
    
    // Allocate host memory
    int *lattice, *lattice_new;
    lattice = (int*) calloc(XSIZE * YSIZE, sizeof(int));
    lattice_new = (int*) calloc(XSIZE * YSIZE, sizeof(int));
      
    // Initialise host array
    srand(time(NULL));
    for (int i = 1; i < XSIZE-1; i++) {
        for (int j = 1; j < YSIZE-1; j++) {
            lattice[i*YSIZE+j] = rand() % 100 < 50;
        }
    }
    
    // Start timer
    clock_t begin = clock();
        
    // Main simulation loop
    for (int i = 0; i <= MAXSTEP; i++) {               
        propagate_and_update(lattice, lattice_new, XSIZE, YSIZE);
    }
      
      
    // Stop timer
    clock_t end = clock();
    
    printf("CPU,%i,%li\n", XSIZE, end-begin);
   
   // Deallocate host memory
   free(lattice);
   free(lattice_new);
}

