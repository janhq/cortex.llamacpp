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

## Build from source

This guide provides step-by-step instructions for building cortex.llamacpp from source on Linux, macOS, and Windows systems.

## Clone the Repository

First, you need to clone the cortex.llamacpp repository:

```bash
git clone --recurse https://github.com/janhq/cortex.llamacpp.git
```

If you don't have git, you can download the source code as a file archive from [cortex.llamacpp GitHub](https://github.com/janhq/cortex.llamacpp). 

## Build library with server example
- **On Windows**
  Install choco
  Install make
  ```
  choco install make -y
  ```

- **On Linux, and Windows:**

  ```bash
  make build-example-server CMAKE_EXTRA_FLAGS=""
  ```
  
- **On MacOS with Apple Silicon:**

  ```bash
  make build-example-server CMAKE_EXTRA_FLAGS="-DLLAMA_METAL_EMBED_LIBRARY=ON"
  ```

- **On MacOS with Intel processors:**

  ```bash
  make build-example-server CMAKE_EXTRA_FLAGS="-DLLAMA_METAL=OFF"
  ```

- **On Linux with CUDA:**

  ```bash
  make build-example-server CMAKE_EXTRA_FLAGS="-DLLAMA_CUDA=ON"
  ```

## Start process

Finally, let's start Server.
- **On MacOS and Linux:**

  ```bash
  mkdir engines\cortex.llamacpp\
  cp ../../build/libengine.dylib engines/cortex.llamacpp/
  ./server
  ```

- **On MacOS and Linux:**

  ```bash
  mkdir engines\cortex.llamacpp\
  cp ../../build/libengine.so engines/cortex.llamacpp/
  ./server
  ```

- **On Windows:**

  ```bash
  cd Release
  mkdir engines\cortex.llamacpp\
  copy ..\..\..\build\Release\engine.dll engines\cortex.llamacpp\
  server.exe
  ```
# Quickstart
// TODO