SHELL := /bin/bash

run:
	@./life_cpu
	@./life_cpu
	@./life_cpu
	@./life_cpu
	@./life_cpu
	@./life_gpu
	@./life_gpu
	@./life_gpu
	@./life_gpu
	@./life_gpu

all:
	nvcc life_gpu.cu -o life_gpu
	gcc life_cpu.c -o life_cpu
