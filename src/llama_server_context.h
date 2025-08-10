#pragma once
#include <atomic>
#include <condition_variable>
#include <mutex>
#include <set>
#include <string>
#include <thread>
#include <vector>

#include "log.h"

// External

#include "src/llama-model.h"
#include "llama_client_slot.h"

#if defined(_WIN32)
#define NOMINMAX
#undef max
#undef min
#endif

using json = nlohmann::json;

#define DEFAULT_OAICOMPAT_MODEL "gpt-3.5-turbo-0613"

struct ServerParams {
  std::string hostname = "127.0.0.1";
  std::string api_key;
  std::string public_path = "examples/server/public";
  int32_t port = 8080;
  int32_t read_timeout = 600;
  int32_t write_timeout = 600;
};

enum class TaskType : uint8_t { kCompletionTask, kCancelTask };

struct TaskServer {
  int id;
  int target_id;
  TaskType type;
  json data;
  bool infill_mode = false;
  bool embedding_mode = false;
  int multitask_id = -1;
};

struct TaskResult {
  int id;
  int multitask_id = -1;
  bool stop;
  bool error;
  json result_json;
};

struct TaskMulti {
  int id;
  std::set<int> subtasks_remaining{};
  std::vector<TaskResult> results{};
};

// completion token output with probabilities

enum class StopType : uint8_t {
  kStopFull,
  kStopPartial,
};

enum class ModelType : uint8_t { kLlm = 0, kEmbedding };

// TODO: reuse llama_detokenize
template <class Iter>
static std::string tokens_to_str(llama_context* ctx, Iter begin, Iter end) {
  std::string ret;
  for (; begin != end; ++begin) {
    ret += common_token_to_piece(ctx, *begin);
  }
  return ret;
}

static void server_log(const char* level, const char* function, int line,
                       const char* message,
                       const nlohmann::ordered_json& extra) {
  nlohmann::ordered_json log{
      {"timestamp", time(nullptr)}, {"level", level},
      {"function", function},       {"line", line},
      {"message", message},
  };

  if (!extra.empty()) {
    log.merge_patch(extra);
  }

  const std::string str =
      log.dump(-1, ' ', false, json::error_handler_t::replace);
  printf("%.*s\n", (int)str.size(), str.data());
  fflush(stdout);
}

template <typename T>
static T json_value(const json& body, const std::string& key,
                    const T& default_value) {
  // Fallback null to default value
  return body.contains(key) && !body.at(key).is_null()
             ? body.value(key, default_value)
             : default_value;
}

struct LlamaServerContext {
  common_init_result llama_init;

  llama_model* model = nullptr;
  llama_context* ctx = nullptr;

  const llama_vocab* vocab = nullptr;

  clip_ctx* clp_ctx = nullptr;

  common_params params;

  llama_batch batch;

  bool multimodal = false;
  bool clean_kv_cache = true;
  bool all_slots_are_idle = false;
  bool add_bos_token = true;
  bool has_eos_token = false;

  std::atomic<int32_t> id_gen;
  int32_t n_ctx;  // total context for all clients / slots

  // Internal
  std::atomic<bool> model_loaded_external = false;

  // system prompt
  bool system_need_update = false;

  std::string system_prompt;
  std::vector<llama_token> system_tokens;

  std::string name_user;  // this should be the antiprompt
  std::string name_assistant;

  // slots / clients
  std::vector<LlamaClientSlot> slots;

  std::vector<TaskServer> queue_tasks;
  std::vector<TaskResult> queue_results;
  std::vector<TaskMulti> queue_multitasks;
  std::mutex mutex_tasks;  // also guards queue_multitasks
  std::condition_variable condition_tasks;
  std::mutex mutex_results;
  std::condition_variable condition_results;
  std::thread bgr_thread;
  ModelType model_type = ModelType::kLlm;

  ~LlamaServerContext();

 public:
  bool LoadModel(const common_params& params_);
  void Initialize();
  void KvCacheClear();
  json GetModelProps();
  int RequestCompletion(json data, bool infill, bool embedding,
                        int multitask_id);
  TaskResult NextResult(int task_id);
  void RequestCancel(int task_id);

  void ReleaseResources();

 private:
  std::vector<llama_token> Tokenize(const json& json_prompt, bool add_special,
                                    bool parse_special) const;

  LlamaClientSlot* GetSlot(int id);

  bool LaunchSlotWithData(LlamaClientSlot*& slot, json data);

  void UpdateSystemPrompt();

  void NotifySystemPromptChanged();

  void ProcessSystemPromptData(const json& sys_props);

  size_t FindStoppingStrings(const std::string& text,
                             const size_t last_token_size, const StopType type,
                             LlamaClientSlot& slot);

  bool ProcessToken(CompletionTokenOutput& result, LlamaClientSlot& slot);

  bool ProcessImages(LlamaClientSlot& slot) const;

  void SendError(TaskServer& task, std::string error);
  void SendError(LlamaClientSlot& slot, const std::string& error);
  void SendError(int id_task, int id_multi, const std::string& error);

  void AddMultiTask(int id, std::vector<int>& sub_ids);

  void UpdateMultiTask(int multitask_id, int subtask_id, TaskResult& result);

  json GetFormatedGeneration(LlamaClientSlot& slot);

  void SendPartialResponse(LlamaClientSlot& slot, CompletionTokenOutput tkn);

  void SendFinalResponse(LlamaClientSlot& slot);

  void SendEmbedding(LlamaClientSlot& slot);

  // for multiple images processing
  bool IngestImages(LlamaClientSlot& slot, int n_batch);

  int SplitMultipromptTask(TaskServer& multiprompt_task);

  void ProcessTasks();

  void DoBackgroundTasks();

  bool UpdateSlots();
};
