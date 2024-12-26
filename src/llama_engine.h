#pragma once

#include <trantor/utils/AsyncFileLogger.h>
#include <unordered_set>
#include "chat_completion_request.h"
#include "cortex-common/enginei.h"
#include "file_logger.h"
#include "llama.h"
#include "llama_server_context.h"
#include "trantor/utils/ConcurrentTaskQueue.h"
#include "trantor/utils/Logger.h"

class LlamaEngine : public EngineI {
 public:
  constexpr static auto kEngineName = "cortex.llamacpp";

  LlamaEngine(int log_option = 0);

  ~LlamaEngine() final;

  // #### Interface ####
  void Load(EngineLoadOption opts) final;

  void Unload(EngineUnloadOption opts) final;

  void HandleChatCompletion(
      std::shared_ptr<Json::Value> jsonBody,
      std::function<void(Json::Value&&, Json::Value&&)>&& callback) final;
  void HandleEmbedding(
      std::shared_ptr<Json::Value> jsonBody,
      std::function<void(Json::Value&&, Json::Value&&)>&& callback) final;
  void LoadModel(
      std::shared_ptr<Json::Value> jsonBody,
      std::function<void(Json::Value&&, Json::Value&&)>&& callback) final;
  void UnloadModel(
      std::shared_ptr<Json::Value> jsonBody,
      std::function<void(Json::Value&&, Json::Value&&)>&& callback) final;
  void GetModelStatus(
      std::shared_ptr<Json::Value> jsonBody,
      std::function<void(Json::Value&&, Json::Value&&)>&& callback) final;
  void GetModels(
      std::shared_ptr<Json::Value> jsonBody,
      std::function<void(Json::Value&&, Json::Value&&)>&& callback) final;
  void SetFileLogger(int max_log_lines, const std::string& log_path) final;
  void SetLogLevel(trantor::Logger::LogLevel log_level =
                       trantor::Logger::LogLevel::kInfo) final;
  void StopInferencing(const std::string& model_id) final;

 private:
  bool LoadModelImpl(std::shared_ptr<Json::Value> jsonBody);
  void HandleInferenceImpl(
      llama::inferences::ChatCompletionRequest&& completion,
      std::function<void(Json::Value&&, Json::Value&&)>&& callback);
  void HandleEmbeddingImpl(
      std::shared_ptr<Json::Value> jsonBody,
      std::function<void(Json::Value&&, Json::Value&&)>&& callback);
  bool CheckModelLoaded(
      std::function<void(Json::Value&&, Json::Value&&)>& callback,
      const std::string& model_id);
  void WarmUpModel(const std::string& model_id);
  bool ShouldInitBackend() const;

  void AddForceStopInferenceModel(const std::string& id);
  void RemoveForceStopInferenceModel(const std::string& id);
  bool HasForceStopInferenceModel(const std::string& id) const;

  bool SpawnLlamaServer(const Json::Value& json_params);
  std::string ConvertJsonToParams(const Json::Value& root);

  bool HandleLlamaCppChatCompletion(
      std::shared_ptr<Json::Value> json_body,
      std::function<void(Json::Value&&, Json::Value&&)>&& callback,
      const std::string& model);

  bool HandleLlamaCppEmbedding(
      std::shared_ptr<Json::Value> json_body,
      std::function<void(Json::Value&&, Json::Value&&)>&& callback,
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
    std::string host;
    int port;
#if defined(_WIN32) || defined(_WIN64)
    PROCESS_INFORMATION pi;
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
  std::unique_ptr<trantor::FileLogger> async_file_logger_;

#if defined(_WIN32)
  std::vector<DLL_DIRECTORY_COOKIE> cookies_;
#endif
};
