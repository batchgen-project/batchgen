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

## License

See the LICENSE file for details.