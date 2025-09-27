# Buddhabrot Fractal Generator

This project is a CUDA-accelerated generator for Buddhabrot fractals.

## Description

The Buddhabrot is a particular rendering of the Mandelbrot set. This project uses CUDA to parallelize the computation and generate high-resolution visualizations of the Buddhabrot fractal. The output is a grayscale image representing the paths of escaping points.

## Prerequisites

To build and run this project, you will need:

*   NVIDIA CUDA Toolkit (`nvcc`) (Tested with CUDA 13.0, Compute Capability 8.6)
*   `make run`

## Getting Started

### Build

To compile the project, run the following command:

```bash
make
```

This will create an executable file named `buddhabrot`.

### Run

To build and run the generator:

```bash
make run
```

The program will generate two files:

*   `buddhabrot.pgm`: A PGM image of the fractal.
*   `histogram.bin`: A binary file containing the raw histogram data.

### Clean

To remove the compiled executable:

```bash
make clean
```

## Roadmap

Here are some of the planned features and improvements:

*   [ ] Implement importance sampling to focus on more interesting regions.
*   [ ] Add support for different color maps to visualize the fractal.

## Gallery
![buddhabrot](images/buddhabrot_copper.png)
![buddhabrot](images/buddhabrot_managua.png)
![buddhabrot](images/buddhabrot_twilight.png)
![buddhabrot](images/buddhabrot_vanimo.png)
