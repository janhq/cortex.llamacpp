#pragma once

#include <trantor/utils/AsyncFileLogger.h>
#include <unordered_set>
#include "chat_completion_request.h"
#include "cortex-common/enginei.h"
#include "file_logger.h"
#include "llama.h"
#include "llama_data.h"
#include "llama_server_context.h"
#include "trantor/utils/ConcurrentTaskQueue.h"
#include "trantor/utils/Logger.h"

using http_callback = std::function<void(Json::Value&&, Json::Value&&)>;

class LlamaEngine : public EngineI {
 public:
  constexpr static auto kEngineName = "cortex.llamacpp";

  LlamaEngine(int log_option = 0);

  ~LlamaEngine() final;

  // Load the engine with the specified options.
  void Load(EngineLoadOption opts) final;

  // Unload the engine with the specified options.
  void Unload(EngineUnloadOption opts) final;

  // Handle a chat completion request with the provided JSON body and callback.
  void HandleChatCompletion(std::shared_ptr<Json::Value> json_body,
                            http_callback&& callback) final;

  // Handle an embedding request with the provided JSON body and callback.
  void HandleEmbedding(std::shared_ptr<Json::Value> json_body,
                       http_callback&& callback) final;

  // Load a model with the provided JSON body and callback.
  void LoadModel(std::shared_ptr<Json::Value> json_body,
                 http_callback&& callback) final;

  // Unload a model with the provided JSON body and callback.
  void UnloadModel(std::shared_ptr<Json::Value> json_body,
                   http_callback&& callback) final;

  // Get the status of a model with the provided JSON body and callback.
  void GetModelStatus(std::shared_ptr<Json::Value> json_body,
                      http_callback&& callback) final;

  // Get the list of available models with the provided JSON body and callback.
  void GetModels(std::shared_ptr<Json::Value> json_body,
                 http_callback&& callback) final;

  // Set the file logger with the maximum number of log lines and log file path.
  void SetFileLogger(int max_log_lines, const std::string& log_path) final;

  // Set the log level for the engine.
  void SetLogLevel(trantor::Logger::LogLevel log_level =
                       trantor::Logger::LogLevel::kInfo) final;

  // Stop the inferencing process for the specified model.
  void StopInferencing(const std::string& model_id) final;


 private:
  bool LoadModelImpl(std::shared_ptr<Json::Value> json_body);
  void HandleInferenceImpl(
      llama::inferences::ChatCompletionRequest&& completion,
      http_callback&& callback);
  void HandleEmbeddingImpl(std::shared_ptr<Json::Value> json_body,
                           http_callback&& callback);
  bool CheckModelLoaded(http_callback& callback, const std::string& model_id);
  void WarmUpModel(const std::string& model_id);
  bool ShouldInitBackend() const;

  void AddForceStopInferenceModel(const std::string& id);
  void RemoveForceStopInferenceModel(const std::string& id);
  bool HasForceStopInferenceModel(const std::string& id) const;

  bool SpawnLlamaServer(const Json::Value& json_params);
  std::string ConvertJsonToParams(const Json::Value& root);
  std::vector<std::string> ConvertJsonToParamsVector(const Json::Value& root);

  bool HandleLlamaCppChatCompletion(std::shared_ptr<Json::Value> json_body,
                                    http_callback&& callback,
                                    const std::string& model);

  // Handle an OpenAI chat completion request with the provided JSON body, callback, and model.
  void HandleOpenAiChatCompletion(std::shared_ptr<Json::Value> json_body,
                                  http_callback&& callback,
                                  const std::string& model);

  // Handle a non-OpenAI chat completion request with the provided JSON body, callback, and model.
  void HandleNonOpenAiChatCompletion(std::shared_ptr<Json::Value> json_body,
                                     http_callback&& callback,
                                     const std::string& model);

  // Handle a LLaMA C++ embedding request with the provided JSON body, callback, and model.
  bool HandleLlamaCppEmbedding(std::shared_ptr<Json::Value> json_body,
                               http_callback&& callback,
                               const std::string& model);

  bool IsLlamaServerModel(const std::string& model) const;

 private:
  struct ServerInfo {
    LlamaServerContext ctx;
    std::unique_ptr<trantor::ConcurrentTaskQueue> q;
    std::string user_prompt;
    std::string ai_prompt;
    std::string system_prompt;
    std::string pre_prompt;
    int repeat_last_n;
    bool caching_enabled;
    std::string grammar_file_content;
    uint64_t start_time;
    uint32_t vram;
    uint32_t dram;
    Json::Value stop_words;
  };

  struct ServerConfig {
    std::unique_ptr<trantor::ConcurrentTaskQueue> q;
    std::string user_prompt;
    std::string ai_prompt;
    std::string system_prompt;
    std::string pre_prompt;
    std::string host;
    int port;
#if defined(_WIN32) || defined(_WIN64)
    PROCESS_INFORMATION pi;
#else
    pid_t pid;
#endif
  };

  // key: model_id, value: ServerInfo
  std::unordered_map<std::string, ServerInfo> server_map_;
  // TODO(sang) use variant map
  std::unordered_map<std::string, ServerConfig> llama_server_map_;
  // lock the force_stop_inference_models_
  mutable std::mutex fsi_mtx_;
  std::unordered_set<std::string> force_stop_inference_models_;

  std::atomic<int> no_of_requests_ = 0;
  std::atomic<int> no_of_chats_ = 0;

  bool print_version_ = true;

  EngineLoadOption load_opt_;

#if defined(_WIN32)
  std::vector<DLL_DIRECTORY_COOKIE> cookies_;
#endif
};
