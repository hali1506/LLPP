# LLPP VT26 Lab group 1 - Assignment 1

## Select version to execute
### Sequential
The sequential version runs as default if no argument is given or `--seq` is provided when running the program. Since it is sequential the execution will only be done on one thread.

#### Example
`./demo/demo scenario.xml`

and 

`./demo/demo --seq scenario.xml`

are equivalent and will run the sequential version of the code on the demo program.


### OpenMP
The OpenMP version runs when provided with `--omp` when running the program. By default the code will run with same amount of hardware threads available, e.g. when running on a system with 16 threads the program will run with 16 threads. To change this behavior make use of the `OMP_NUM_THREADS` variable when running the program.

#### Example
`./demo/demo --omp scenario.xml`

will run the OpenMP version of the code on the demo program.


### C++ Threads
The C++ Threads version runs when provided with `--cppthreads` when running the program. y default the code will run with same amount of hardware threads available, e.g. when running on a system with 16 threads the program will run with 16 threads. To change this behavior change the macro on row 28 in `ped_model.cpp` to the number of threads desired.

#### Example
`./demo/demo --cppthreads scenario.xml`

will run the C++ Threads version of the code on the demo program.

### SIMD
The SIMD version runs when provided with `--simd` when running the program.

#### Example
`./demo/demo --simd scenario.xml`

will run the simd version of the code on the demo program.

### CUDA
The CUDA version runs when provided with `--cuda` when running the program.

#### Example
`./demo/demo --cuda scenario.xml`

will run the cuda version of the code on the demo program.

## Machine specifications
The experiments were done on the Crocodile server using the slurm scheduler by running using the `srun` command.

### Hardware specifications
AMD Ryzen 7950X3D 16-Core processor with SMT for a total of 32-threads, but limited to 16-threads when running using `srun`.

The RAM on the server is limited to 2GB with unknown speed.

The GPU available on Crocodile is a NVIDIA GeForce RTX 4080 SUPER with 10240 CUDA cores


## Contribution
The design idea was discussed within the group but mainly implemented by William. Felix had main responsibility of the report while Hanna contributed to the SIMD implementation and worked on the report