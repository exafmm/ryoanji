
# Ryoanji - a distributed N-body solver for AMD and NVIDIA GPUs

Ryoanji is a Barnes-Hut N-body solver for gravity and electrostatics.
It employs [EXAFMM](https://github.com/exafmm/exafmm) multipole kernels and a Barnes-Hut tree-traversal
algorithm inspired by [Bonsai](https://github.com/treecode/Bonsai). Octrees and domain decomposition are
handled by [Cornerstone Octree](https://github.com/sekelle/cornerstone-octree), see Ref. [1].

Ryoanji is optimized to run efficiently on both AMD and NVIDIA GPUs, though a CPU implementation is provided as well.

## Folder structure

```
Ryoanji.git
├── README.md
├── cstone          - Cornerstone library: octree building and domain decomposition
│                     (git subtree of https://github.com/sekelle/cornerstone-octree)
│                             
└── ryoanji                            - Ryoanji: N-body solver
   ├── src
   └── test
       ├── demo.cu                     - single-rank demonstrator app
       ├── demo_mpi.cpp                - multi-rank demonstrator app
       ├── interface
       │   └─── global_forces_gpu.cpp   - multi-rank correctness check vs. direct sum
       ├── nbody
       └── test_main.cpp
```


## Compilation

Ryoanji is written in C++ and CUDA. The host `.cpp` translation units require a C++20 compiler
(GCC 11 and later, Clang 14 and later), while `.cu` translation units are compiled in the C++17 standard.
CUDA version: 11.6 or later, HIP version 5.2 or later.

NVIDIA CUDA, A100
```bash
CC=mpicc CXX=mpicxx cmake -DCMAKE_CUDA_ARCHITECTURES=80 -DCMAKE_CUDA_FLAGS=-ccbin=mpicxx -DGPU_DIRECT=<ON/OFF> <GIT_SOURCE_DIR>
make -j
```

AMD HIP, MI250x

The code can directly be built with HIP, no hipification needed:

```bash
CC=mpicc CXX=mpicxx cmake -DCMAKE_HIP_ARCHITECTURES=gfx90a -DCSTONE_WITH_GPU_AWARE_MPI=<ON/OFF> <GIT_SOURCE_DIR> && make -j
```

## Performance

One particle-particle (P2P) interaction counts as 23 flops, a multipole-particle (M2P) interaction with
spherical hexadecapoles `(P=4)` counts as `2 * P^3 = 128` flops. The performance numbers given below only
take P2P and M2P into account. Additional floating point operations due to tree node evaluations
(multipole acceptance criteria, MAC) or warp-padding overheads are not taken into account.
The opening angle `theta` was set to `0.5`.


* 1 x NVIDIA A100: 10.4 TFlop/s (FP32) per GPU, 62.2 million particles / second per GPU, 67 million particles total
* 4 x NVIDIA A100: 10.9 TFlop/s (FP32) per GPU, 35.5 million particles / second per GPU, 3 billion particles total


* 1x AMD MI250X: 15.1 TFlops/s (FP32) per GPU (2 GCDs), 60.0 million particles / second per GPU, 67 million particles total
* 4x AMD MI250X: 15.4 TFlops/s (FP32) per GPU (2 GCDs), 50.0 million particles / second per GPU, 3 billion particles total


* 8208x AMD MI250X (LUMI-G): ~107 PFlops/s (FP64), 44.3 million particles / second per GPU (2 GCDs), 8 trillion particles total (in 22.2 seconds) [1]


Note: the multi-rank demonstrator app provided here initializes random particles on all MPI ranks for the same spatial domain.
This requires all-to-all communication to construct the sub-domains of each rank and is not feasible for large number of ranks.
In order to construct domains for trillions of particles such as in Ref. [1], optimized initialization strategies are required
that places particles into the correct sub-domains. This is possible for Space-Filling-Curve (SFC) sorted input files
or for in-situ initialization for particle ensembles with known (density) distribution functions.
An application front-end that implements this capability, in addition to I/O and a time-stepping loop is
available as part of the [SPH-EXA project](https://github.com/unibas-dmi-hpc/SPH-EXA).

## Accuracy and correctness

The demonstrator apps are configured by default to use an opening angle of `theta = 0.5` cartesian quadrupole expansions.
This yields a 1st-percentile error of `~5e-4` in the accelerations.

```
$$$ mpiexec -np 8 ./interface/global_forces_gpu 
rank 0 1st-percentile acc error 0.000410922, max acc error 0.00267019
rank 1 1st-percentile acc error 0.000501579, max acc error 0.00327092
rank 2 1st-percentile acc error 0.000362208, max acc error 0.00280561
rank 3 1st-percentile acc error 0.000481996, max acc error 0.0251728
rank 4 1st-percentile acc error 0.000579059, max acc error 0.0110242
rank 5 1st-percentile acc error 0.000442119, max acc error 0.00426394
rank 6 1st-percentile acc error 0.000470549, max acc error 0.00187002
rank 7 1st-percentile acc error 0.000527458, max acc error 0.00332407
global reference potential -0.706933, BH global potential -0.706931
```

## References

[1] [S. Keller et al. 2023, Cornerstone: Octree Construction Algorithms for Scalable Particle Simulations](https://doi.org/10.1145/3592979.3593417)

## Authors

* **Sebastian Keller**
* **Rio Yokota**
