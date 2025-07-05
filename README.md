# cuda-raytracer
A simple path tracer written in CUDA.

## Setup
1. Use the nix flake development environment by running `nix develop`. If you do not have Nix setup, read `flake.nix` to see what you need.
2. In the `src/` directory, run `mkdir build && cd build && cmake .. && make`. You will then have a binary you can run.
