.PHONY: all clean

all: main.cpp include/buddhabrot.cu
	nvcc main.cpp include/buddhabrot.cu -o buddhabrot.out
clean:
	rm buddhabrot.out -f
run: all
	./buddhabrot
