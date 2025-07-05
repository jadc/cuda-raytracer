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
                    ];

                    shellHook = ''
                        export CUDA_PATH=${pkgs.cudaPackages.cudatoolkit}
                        export PATH=$CUDA_PATH/bin:$PATH
                        export LIBRARY_PATH=$CUDA_PATH/lib:$LIBRARY_PATH
                        export LD_LIBRARY_PATH=$CUDA_PATH/lib:/run/opengl-driver/lib:/run/opengl-driver-32
                    '';
                };
            });
}
