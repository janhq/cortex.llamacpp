# cortex.llamacpp
cortex.llamacpp is a high-efficiency C++ inference engine for edge computing.

It is a dynamic library that can be loaded by any server at runtime.

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
  make build-example-server CMAKE_EXTRA_FLAGS="-DGGML_METAL_EMBED_LIBRARY=ON"
  ```

- **On MacOS with Intel processors:**

  ```bash
  make build-example-server CMAKE_EXTRA_FLAGS="-DGGML_METAL=OFF"
  ```

- **On Linux with CUDA:**

  ```bash
  make build-example-server CMAKE_EXTRA_FLAGS="-DGGML_CUDA=ON"
  ```

# Quickstart
**Step 1: Downloading a Model**

```bash
mkdir model && cd model
wget -O llama-2-7b-model.gguf https://huggingface.co/TheBloke/Llama-2-7B-Chat-GGUF/resolve/main/llama-2-7b-chat.Q5_K_M.gguf?download=true
```

**Step 2: Start server**
- **On MacOS:**

  ```bash
  mkdir engines\cortex.llamacpp\
  cp ../../build/libengine.dylib engines/cortex.llamacpp/
  ./server
  ```

- **On Linux:**

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

**Step 3: Load model**
```bash title="Load model"
curl http://localhost:3928/loadmodel \
  -H 'Content-Type: application/json' \
  -d '{
    "llama_model_path": "/model/llama-2-7b-model.gguf",
    "model_alias": "llama-2-7b-model",
    "ctx_len": 512,
    "ngl": 100,
    "model_type": "llm"
  }'
```
**Step 4: Making an Inference**

```bash title="cortex-cpp Inference"
curl http://localhost:3928/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "messages": [
      {
        "role": "user",
        "content": "Who won the world series in 2020?"
      },
    ],
    "model": "llama-2-7b-model"
  }'
```

Table of parameters

| Parameter        | Type    | Description                                                  |
|------------------|---------|--------------------------------------------------------------|
| `llama_model_path` | String  | The file path to the LLaMA model.                            |
| `ngl`              | Integer | The number of GPU layers to use.                             |
| `ctx_len`          | Integer | The context length for the model operations.                 |
| `embedding`        | Boolean | Whether to use embedding in the model.                       |
| `n_parallel`       | Integer | The number of parallel operations. |
| `cont_batching`    | Boolean | Whether to use continuous batching.                          |
| `user_prompt`      | String  | The prompt to use for the user.                              |
| `ai_prompt`        | String  | The prompt to use for the AI assistant.                      |
| `system_prompt`    | String  | The prompt to use for system rules.                          |
| `pre_prompt`    | String  | The prompt to use for internal configuration.                          |
| `cpu_threads`   | Integer | The number of threads to use for inferencing (CPU MODE ONLY) |
| `n_batch`       | Integer | The batch size for prompt eval step |
| `caching_enabled` | Boolean | To enable prompt caching or not   |
|`grp_attn_n`|Integer|Group attention factor in self-extend|
|`grp_attn_w`|Integer|Group attention width in self-extend|
|`mlock`|Boolean|Prevent system swapping of the model to disk in macOS|
|`grammar_file`| String |You can constrain the sampling using GBNF grammars by providing path to a grammar file|
|`model_type` | String | Model type we want to use: llm or embedding, default value is llm|
|`model_alias`| String | Used as model_id if specified in request, mandatory in loadmodel|
|`model`      | String | Used as model_id if specified in request, mandatory in chat/embedding request|
|`flash_attn` | Boolean| To enable Flash Attention, default is true|
|`cache_type` | String| KV cache type: f16, q8_0, q4_0, default is f16|
|`use_mmap` | Boolean| To enable mmap, default is true|
