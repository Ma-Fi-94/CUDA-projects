SHELL := /bin/bash

gpu:
	for size in 10000 ; do \
		./life_gpu $$size $$size ; \
		./life_gpu $$size $$size ; \
		./life_gpu $$size $$size ; \
		./life_gpu $$size $$size ; \
		./life_gpu $$size $$size ; \
		./life_gpu $$size $$size ; \
		./life_gpu $$size $$size ; \
		./life_gpu $$size $$size ; \
		./life_gpu $$size $$size ; \
		./life_gpu $$size $$size ; \
	done

cpu:
	for size in 10 20 50 100 200 500 1000 ; do \
		./life_cpu $$size $$size ; \
		./life_cpu $$size $$size ; \
		./life_cpu $$size $$size ; \
		./life_cpu $$size $$size ; \
		./life_cpu $$size $$size ; \
	done

all:
	nvcc life_gpu.cu -o life_gpu
	gcc life_cpu.c -o life_cpu
