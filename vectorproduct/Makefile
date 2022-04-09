run:
	nvcc vector_prod_singleblock.cu -o vector_prod_singleblock
	./vector_prod_singleblock

all:
	nvcc vector_prod_singleblock.cu -o vector_prod_singleblock

profile: all
	sudo $$(which nvprof) ./vector_prod_singleblock
