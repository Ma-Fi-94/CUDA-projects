run:
	nvcc life2.cu -o life2
	./life2

all:
	nvcc life2.cu -o life2

profile: all
	sudo $$(which nvprof) ./life2
