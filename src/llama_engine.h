#pragma once
#include <trantor/utils/AsyncFileLogger.h>
#include "chat_completion_request.h"
#include "cortex-common/enginei.h"
#include "file_logger.h"
#include "llama.h"
#include "llama_server_context.h"
#include "trantor/utils/ConcurrentTaskQueue.h"

class LlamaEngine : public EngineI {
 public:
  LlamaEngine(int log_option = 0);
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
  void SetFileLogger(int max_log_lines, const std::string& log_path) final;
  void SetLoggerOption(const Json::Value& json_body);

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
  std::unique_ptr<trantor::FileLogger> asynce_file_logger_;
};