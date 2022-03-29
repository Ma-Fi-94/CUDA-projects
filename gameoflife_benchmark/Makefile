SHELL := /bin/bash

run:
	./life_cpu
	./life_gpu

all:
	nvcc life_gpu.cu -o life_gpu
	clang life_cpu.c -o life_cpu
