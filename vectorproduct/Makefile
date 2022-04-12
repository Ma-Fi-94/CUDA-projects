run:
	nvcc vector_prod_multiblock.cu -o vector_prod_multiblock
	./vector_prod_multiblock

all:
	nvcc vector_prod_multiblock.cu -o vector_prod_multiblock

profile: all
	sudo $$(which nvprof) ./vector_prod_multiblock
