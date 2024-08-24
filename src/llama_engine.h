#pragma once
#include "chat_completion_request.h"
#include "cortex-common/enginei.h"
#include "llama_server_context.h"
#include "trantor/utils/ConcurrentTaskQueue.h"
#include "llama.h"
#include <trantor/utils/AsyncFileLogger.h>

constexpr char log_base_name[] = "logs/cortex";
constexpr char log_folder[] = "logs";
constexpr size_t max_log_file_size = 20000000; // ~20mb

class LlamaEngine : public EngineI {
 public:
  LlamaEngine();
  ~LlamaEngine() final;
  // #### Interface ####
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
  void SetFileLogger();

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

  // key: model_id, value: ServerInfo
  std::unordered_map<std::string, ServerInfo> server_map_;

  std::atomic<int> no_of_requests_ = 0;
  std::atomic<int> no_of_chats_ = 0;

  bool print_version_ = true;
  std::unique_ptr<trantor::AsyncFileLogger> asynce_file_logger_;
};