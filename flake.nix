{
    # This flake assumes you already have NVIDIA and CUDA drivers installed.

    inputs = {
        nixpkgs.url = "nixpkgs/nixpkgs-unstable";
        flake-utils.url = "github:numtide/flake-utils";
    };

    outputs = { self, nixpkgs, flake-utils }:
        flake-utils.lib.eachDefaultSystem (system:
            let
                pkgs = import nixpkgs {
                    inherit system;
                    config.allowUnfree = true;
                };
            in {
                devShells.default = pkgs.mkShell {
                    buildInputs = with pkgs; [
                        cmake
                        cudaPackages.cuda_cudart
                        cudaPackages.cuda_nvcc
                        cudaPackages.cudatoolkit
                        gcc13
                    ];

                    shellHook = ''
                        export CUDA_PATH=${pkgs.cudatoolkit}
                        export CC=${pkgs.gcc13}/bin/gcc
                        export CXX=${pkgs.gcc13}/bin/g++
                        export PATH=${pkgs.gcc13}/bin:$PATH
                        export LD_LIBRARY_PATH=${
                            pkgs.lib.makeLibraryPath [
                                "/run/opengl-driver"
                                pkgs.cudaPackages.cudatoolkit
                                pkgs.cudaPackages.cudnn
                            ]
                        }:$LD_LIBRARY_PATH
                        export LIBRARY_PATH=${
                            pkgs.lib.makeLibraryPath [
                                pkgs.cudaPackages.cudatoolkit
                            ]
                        }:$LIBRARY_PATH
                    '';
                };
            });
}
