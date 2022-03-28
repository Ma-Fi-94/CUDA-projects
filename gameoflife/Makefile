run:
	nvcc life.cu -o life
	./life

all:
	nvcc life.cu -o life

profile: all
	sudo $$(which nvprof) ./life
