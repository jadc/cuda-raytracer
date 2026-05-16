# cuda-raytracer
A simple path tracer written in CUDA.

## Setup
1. Use the nix flake development environment by running `nix develop`. If you do not have Nix setup, read `flake.nix` to see what you need.
2. In the `src/` directory, run `mkdir build && cd build && cmake .. && make`. You will then have a binary you can run.

## Progress
These are some screenshots from throughout development.

#### Surface normals visualization
![](img/2026-05-06-01.png)

#### Surface normals with anti-aliasing
![](img/2026-05-06-02.png)

#### Uniform diffuse lighting, with a bug
![](img/2026-05-14-01.png)

#### Fixed bug
![](img/2026-05-14-02.png)

#### Lambertian reflections
![](img/2026-05-14-03.png)

#### Broke the ppm export logic
![](img/2026-05-14-04b.png)

#### Lambertian reflections with gamma correction
![](img/2026-05-14-04.png)

#### Material system with Lambertian and metal
![](img/2026-05-16-01.png)
