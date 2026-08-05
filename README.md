# Orchestra

Orchestra is a set of high-level abstractions for heterogeneous programming based on the [OCL-PolyHok](https://github.com/Equiel-1703/ocl-polyhok) DSL. It provides a way for developers to write compute kernels that can be executed on CPUs and GPUs without having to change their code or deal with the low-level aspects of OpenCL, such as manual memory management, data typing, and kernel compilation.

## Prerequisites

To get started with Orchestra, first ensure you have the following prerequisites on your system:

- **Elixir 1.17 with Erlang/OTP 27 or higher**. We recommend using [asdf](https://asdf-vm.com/) to manage Elixir and Erlang versions.

- **Erlang development libraries**. On Debian/Ubuntu systems, you can install them using `apt`:

  ```bash
  sudo apt install erlang-dev
  ```

- **C++17 compatible compiler**. You can use GCC or Clang. For Debian/Ubuntu systems, install GCC with:

  ```bash
  sudo apt install build-essential
  ```

- **CMake 3.12 or higher**. You can install it using your package manager (such as `apt`) or download it directly from the [CMake website](https://cmake.org/download/).

- **OpenCL 3.0 compatible hardware**. Check your CPU and GPU specifications to ensure support for OpenCL 3.0. We highly recommend installing `clinfo` to verify the OpenCL platforms and devices available on your system. For Debian/Ubuntu systems:

  ```bash
  sudo apt install clinfo
  ```

- **OpenCL generic ICD loader and C/C++ headers**. Install the necessary OpenCL development packages. For Debian/Ubuntu systems:

  ```bash
  sudo apt install ocl-icd-opencl-dev opencl-c-headers opencl-clhpp-headers
  ```

  **Note on the ICD Loader**: The packages listed above provide the development headers and the OpenCL Installable Client Driver (ICD) loader. The ICD loader acts as a dispatcher/multiplexer, meaning it does not contain hardware drivers itself, but allows multiple OpenCL implementations (e.g., NVIDIA, AMD, PoCL) to coexist on the same system. Therefore, you must install the correct driver for your CPU and GPU so the ICD loader can discover it at runtime and route the OpenCL API calls to the correct hardware.

## Hardware-Specific Recommendations

Orchestra was developed and tested on two primary setups. Here is what we recommend based on our testing:

### AMD APU Systems (e.g., Ryzen 5 5500U with AMD Lucienne iGPU)

If you have a system with an AMD iGPU (like our Lenovo IdeaPad 3 test machine), we strongly recommend using the **Mesa OpenCL driver** for GPU parallelism in Orchestra. Mesa drivers works very well with AMD's iGPUs. We also recommend enabling `rusticl` (a modern OpenCL implementation written in Rust). To install the Mesa OpenCL driver in Debian/Ubuntu systems, run:

```bash
sudo apt install mesa-opencl-icd
```

To enable `rusticl`, set the following environment variable in your shell configuration file (e.g., `~/.bashrc` or `~/.zshrc`):

```bash
export RUSTICL_ENABLE='radeonsi'
```

### Discrete NVIDIA GPUs (e.g. RTX 4070)

If you have a dedicated NVIDIA GPU, use the proprietary NVIDIA drivers, as they come bundled with the necessary OpenCL runtime support. Follow [NVIDIA's driver installation guide](https://docs.nvidia.com/datacenter/tesla/driver-installation-guide/latest/index.html) for detailed instructions.

### For CPU Parallelism (All Systems)

Regardless of your GPU, we recommend using the `pocl` (Portable Computing Language) implementation for CPU-based OpenCL parallelism. To install PoCL on Debian/Ubuntu systems, run:

```bash
sudo apt install pocl-opencl-icd
```

## Getting Started

Once you have the prerequisites installed, follow these steps to set up and compile Orchestra:

1. **Clone the repository and navigate to the project directory**:

    ```bash
    git clone https://github.com/Equiel-1703/orchestra.git
    cd orchestra
    ```

2. **Fetch the Elixir dependencies**:

    ```bash
    mix deps.get
    ```

3. **Compile the C++ NIFs**. Generate the CMake build scripts and compile the NIFs for Orchestra's OpenCL runtime and BMP generation.

    ```bash
    cmake -S . -B CMake
    cmake --build CMake
    ```

4. **Compile Orchestra**:

    ```bash
    mix compile
    ```

5. **Done!** Orchestra is now ready to use. You can test it by running the included Julia Set program to generate a BMP image of the fractal:

    ```bash
    mix run benchmarks/gpu/julia.exs 1024
    ```

    This program uses your GPU to generate a Julia Set fractal at a resolution of 1024x1024 pixels. The resulting image is saved as `julia_set.bmp` in the project root directory.

## Repository Structure

The repository is organized as follows:

- `lib/`: Core Elixir implementation of Orchestra, containing its macros, functions, tensor operations, contexts, etc.
- `c_src/`: C++/OpenCL NIF runtime source code for CPU and GPU execution.
- `benchmarks/`: Benchmark implementations evaluated in the paper:
  - `benchmarks/cpu/`: Linear algebra benchmarks.
  - `benchmarks/cooperative/bfs`: Breadth-First Search (BFS) graph traversal benchmark and the used datasets. The datasets were obtained from the Stanford Network Analysis Project (SNAP) repository.
- `LICENSE`: MIT License terms.

## Licensing

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Publications

Orchestra was presented in the following paper:

> **Orchestra: CPU-GPU Cooperative Programming in Elixir**  
> Henrique Gabriel Rodrigues, Eduardo Beloni Mailan, Gerson Geraldo H. Cavalheiro, André Rauber Du Bois.  
> Published in the *30th Brazilian Symposium on Programming Languages (SBLP 2026)*.

- **Paper PDF:** [Download Paper PDF](./paper/Orchestra.pdf)