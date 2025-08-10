# MoE Gen Kernel

This project is MoE Gen Kernel, providing CUDA implementations for MoE Model.

## Build Instructions

Make sure Python and required dependencies are installed. Then, run the following command in the project root directory:

```bash
bash build.sh 3.11 12.8
```

After building, a Python wheel file (`.whl`) will be generated for installation and distribution in the `dist` directory.

## Local Build
You can build the project locally by running the `make build` command. For best results, ensure that the following requirements are met:

- CMake version: greater than 3.29


The build artifacts will be placed in the `dist/` directory, and the resulting Python package will be automatically installed into the local environment via pip.

If you encounter build errors, you can use the `make rebuild` command to clear all cached files and perform a clean rebuild.

Steps for developing a kernel:

1. Add a CUDA source file (`.cu`) to the csrc directory.
2. Register the new source file in `CMakeLists.txt` by appending it to the `SOURCES` list:
    ```cmake
    set(SOURCES
        "csrc/xxx.cu"
    )
    ```
3. Extend the C++–Python interface:
    - Define the corresponding Python bindings in `common_extension.cc`.
    - Add the function declaration to the header file `include/mgn_kernel_ops.h`.
4. Integrate the Python-level API by adding the corresponding definitions in the `python/mgn_kernel` directory.


## License

See the LICENSE file for details.