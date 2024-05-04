# cortex.llamacpp
cortex.llamacpp is a high-efficiency C++ inference engine for edge computing.

# Repo Structure
```
.
├── base -> Engine interface
├── examples -> Server example to integrate engine
├── llama.cpp -> Upstream llama C++
├── src -> Engine implementation
├── third-party -> Dependencies of the cortex.llamacpp project
```
# Quickstart
// TODO

## Build from source

This guide provides step-by-step instructions for building cortex.llamacpp from source on Linux, macOS, and Windows systems.

## Clone the Repository

First, you need to clone the cortex.llamacpp repository:

```bash
git clone --recurse https://github.com/janhq/cortex.llamacpp.git
```

If you don't have git, you can download the source code as a file archive from [cortex.llamacpp GitHub](https://github.com/janhq/cortex.llamacpp). 
## Install Dependencies

Next, let's install the necessary dependencies.

- **On MacOS with Apple Silicon:**

  ```bash
  ./install_deps.sh
  ```

- **On Windows:**

  ```bash
  cmake -S ./third-party -B ./build_deps/third-party
  cmake --build ./build_deps/third-party --config Release
  ```

This creates a `build_deps` folder.

## Generate build file

Now, let's generate the build files.

- **On MacOS, Linux, and Windows:**

  ```bash
  mkdir build && cd build
  cmake ..
  ```

- **On MacOS with Intel processors:**

  ```bash
  mkdir build && cd build
  cmake -DLLAMA_METAL=OFF ..
  ```

- **On Linux with CUDA:**

  ```bash
  mkdir build && cd build
  cmake -DLLAMA_CUDA=ON ..
  ```

## Build the library

Time to build Cortex.llamacpp!

- **On MacOS:**

  ```bash
  make -j $(sysctl -n hw.physicalcpu)
  ```

- **On Linux:**

  ```bash
  make -j $(nproc)
  ```

- **On Windows:**

  ```bash
  make -j $(%NUMBER_OF_PROCESSORS%)
  ```

## Build server example

- **On MacOS:**

  ```bash
  cd ..
  mkdir -p examples/server/build
  cd examples/server/build
  cmake ..
  make -j $(sysctl -n hw.physicalcpu)
  ```

- **On Linux:**

  ```bash
  cd ..
  mkdir -p examples/server/build
  cd examples/server/build
  cmake ..
  make -j $(nproc)
  ```

- **On Windows:**

  ```bash
  cd ..
  mkdir  examples\server\build
  cd examples\server\build
  cmake ..
  make -j $(%NUMBER_OF_PROCESSORS%)
  ```

## Start process

Finally, let's start Server.

- **On MacOS and Linux:**

  ```bash
  ./server
  ```

- **On Windows:**

  ```bash
  cd Release
  mkdir engines\cortex.llamacpp\
  copy ..\..\..\build\Release\engine.dll engines\cortex.llamacpp\
  server.exe
  ```
