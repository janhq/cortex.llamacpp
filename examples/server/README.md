This application allows you to perform inference with various AI models, including vision, text, and embedding models. It employs different strategies for serving these models efficiently, tailored to their specific requirements.

- Vision Models:
For vision models, the application utilizes a dedicated, customized server that runs within the same process as the main application. This approach ensures efficient resource utilization and minimizes overhead, providing seamless inference for vision-based tasks.

- Text and Embedding Models:
To handle text and embedding models, the application spawns a separate child process for each model. This isolation technique prevents potential resource contention and ensures optimal performance, as each model operates independently within its dedicated process environment.

By adopting distinct serving strategies based on model types, the application optimizes resource allocation, maximizes performance, and ensures reliable and efficient inference across a diverse range of AI models.

**API Endpoints**

The application provides the following API endpoints for interacting with the models:

**Load Model**

Endpoint: /loadmodel

Method: curl -X POST

Description: Loads a specified model into the application. For text and embedding models, this endpoint will spawn a new child process to serve the model.

```bash title="Load model"
curl --location 'http://localhost:3928/loadmodel' \
--header 'Content-Type: application/json' \
--data '{
    "llama_model_path": "/model/llama-2-7b-model.gguf",
    "model_alias": "llama-2-7b-model",
    "ctx_len": 512,
    "ngl": 100,
    "model_type": "llm"
  }'
```

**Chat Completion**

Endpoint: /v1/chat/completions

Method: curl -X POST

Description: Performs chat completion using a loaded text model.

```bash title="Inference"
curl --location 'http://localhost:3928/v1/chat/completions' \
--header 'Content-Type: application/json' \
--data '{
    "messages": [
      {
        "role": "user",
        "content": "Who won the world series in 2020?"
      },
    ],
    "model": "llama-2-7b-model"
  }'
```

**Embedding**

Endpoint: /v1/embeddings

Method: curl -X POST

Description: Requests an embedding using a loaded embedding model.
```bash title="Embeddings"
curl --location '127.0.0.1:3928/v1/embeddings' \
--header 'Content-Type: application/json' \
--data '{
        "input": ["hello", "The food was delicious and the waiter..."],
        "model":"llama-2-7b-model",
        "encoding_format": "base64"
}'
```

**Unload Model**

Endpoint: /unloadmodel
Method: curl -X POST

Description: Unloads a specified model from the application. For text and embedding models, this endpoint will also stop the associated child process.
```bash title="Unload Model"
curl --location '127.0.0.1:3928/unloadmodel' \
--header 'Content-Type: application/json' \
--data '{
    "model": "test"
}'
```

**Multiple Models**

This application is designed to handle multiple AI models concurrently, ensuring efficient resource utilization and optimal performance. The serving strategy employed varies based on the model type:
- Vision models: multiple instances can run within the same process. 
- Text and embedding models: each model will have its own child process.

**Notes**

For vision models, a customized model is started within the same process to serve the model. No new process is needed.
For text and embedding models, a new child process is spawned to serve each model.