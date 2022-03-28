run:
	nvcc vector_add.cu -o vector_add
	./vector_add

all:
	nvcc vector_add.cu -o vector_add

profile: all
	sudo $$(which nvprof) ./vector_add
