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

using http_callback = std::function<void(Json::Value&&, Json::Value&&)>;

class LlamaEngine : public EngineI {
 public:
  constexpr static auto kEngineName = "cortex.llamacpp";

  LlamaEngine(int log_option = 0);

  ~LlamaEngine() final;

  // #### Interface ####
  void Load(EngineLoadOption opts) final;

  void Unload(EngineUnloadOption opts) final;

  void HandleChatCompletion(std::shared_ptr<Json::Value> jsonBody,
                            http_callback&& callback) final;
  void HandleEmbedding(std::shared_ptr<Json::Value> jsonBody,
                       http_callback&& callback) final;
  void LoadModel(std::shared_ptr<Json::Value> jsonBody,
                 http_callback&& callback) final;
  void UnloadModel(std::shared_ptr<Json::Value> jsonBody,
                   http_callback&& callback) final;
  void GetModelStatus(std::shared_ptr<Json::Value> jsonBody,
                      http_callback&& callback) final;
  void GetModels(std::shared_ptr<Json::Value> jsonBody,
                 http_callback&& callback) final;
  void SetFileLogger(int max_log_lines, const std::string& log_path) final;
  void SetLogLevel(trantor::Logger::LogLevel log_level =
                       trantor::Logger::LogLevel::kInfo) final;
  void StopInferencing(const std::string& model_id) final;

 private:
  bool LoadModelImpl(std::shared_ptr<Json::Value> jsonBody);
  void HandleInferenceImpl(
      llama::inferences::ChatCompletionRequest&& completion,
      http_callback&& callback);
  void HandleEmbeddingImpl(std::shared_ptr<Json::Value> jsonBody,
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

  void HandleOpenAiChatCompletion(std::shared_ptr<Json::Value> json_body,
                                  http_callback&& callback,
                                  const std::string& model);
  void HandleNonOpenAiChatCompletion(std::shared_ptr<Json::Value> json_body,
                                     http_callback&& callback,
                                     const std::string& model);

  bool HandleLlamaCppEmbedding(std::shared_ptr<Json::Value> json_body,
                               http_callback&& callback,
                               const std::string& model);

  bool IsLlamaServerModel(const std::string& model) const;

 private:
  struct IsDone {
    bool is_done;
    int operator()() { return is_done; }
  };
  struct HasError {
    bool has_error;
    int operator()() { return has_error; }
  };
  struct IsStream {
    bool is_stream;
    int operator()() { return is_stream; }
  };
  struct StatusCode {
    int status_code;
    int operator()() { return status_code; }
  };
  struct ResStatus {
   private:
    IsDone is_done;
    HasError has_error;
    IsStream is_stream;
    StatusCode status_code;

   public:
    ResStatus(IsDone is_done, HasError has_error, IsStream is_stream,
              StatusCode status_code)
        : is_done(is_done),
          has_error(has_error),
          is_stream(is_stream),
          status_code(status_code) {}

    Json::Value ToJson() {
      Json::Value status;
      status["is_done"] = is_done();
      status["has_error"] = has_error();
      status["is_stream"] = is_stream();
      status["status_code"] = status_code();
      return status;
    };
  };

  struct ResStreamData {
   private:
    std::string s;

   public:
    ResStreamData(std::string s) : s(std::move(s)) {}
    Json::Value ToJson() {
      Json::Value d;
      d["data"] = s;
      return d;
    }
  };

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
  std::unique_ptr<trantor::FileLogger> async_file_logger_;

#if defined(_WIN32)
  std::vector<DLL_DIRECTORY_COOKIE> cookies_;
#endif
};
