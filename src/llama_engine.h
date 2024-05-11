#pragma once
#include "cortex-common/enginei.h"
#include "llama_server_context.h"
#include "trantor/utils/ConcurrentTaskQueue.h"
#include "chat_completion_request.h"

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

 private:
  bool LoadModelImpl(std::shared_ptr<Json::Value> jsonBody);
  void HandleInferenceImpl(
      llama::inferences::ChatCompletionRequest&& completion,
      std::function<void(Json::Value&&, Json::Value&&)>&& callback);
  void HandleEmbeddingImpl(
      std::shared_ptr<Json::Value> jsonBody,
      std::function<void(Json::Value&&, Json::Value&&)>&& callback);
  bool CheckModelLoaded(
      std::function<void(Json::Value&&, Json::Value&&)>& callback);
  void WarmUpModel();
  void HandleBackgroundTask();
  void StopBackgroundTask();

 private:
  LlamaServerContext llama_;
  std::unique_ptr<trantor::ConcurrentTaskQueue> queue_;
  std::thread bgr_thread_;

  std::string user_prompt_;
  std::string ai_prompt_;
  std::string system_prompt_;
  std::string pre_prompt_;
  int repeat_last_n_;
  bool caching_enabled_;
  std::atomic<int> no_of_requests_ = 0;
  std::atomic<int> no_of_chats_ = 0;
  int clean_cache_threshold_;
  std::string grammar_file_content_;
};