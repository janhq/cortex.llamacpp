#include "llama_engine.h"

#include <chrono>
#include "json/writer.h"
#include "llama_utils.h"
#include "trantor/utils/Logger.h"

namespace {
constexpr const int k200OK = 200;
constexpr const int k400BadRequest = 400;
constexpr const int k409Conflict = 409;
constexpr const int k500InternalServerError = 500;
constexpr const int kFileLoggerOption = 0;

constexpr const auto kTypeF16 = "f16";
constexpr const auto kType_Q8_0 = "q8_0";
constexpr const auto kType_Q4_0 = "q4_0";

bool IsValidCacheType(const std::string& c) {
  if (c != kTypeF16 && c != kType_Q8_0 && c != kType_Q4_0) {
    return false;
  }
  return true;
}

struct InferenceState {
  int task_id;
  LlamaServerContext& llama;
  // Check if we receive the first token, set it to false after receiving
  bool is_first_token = true;

  InferenceState(LlamaServerContext& l) : llama(l) {}
};

/**
 * This function is to create the smart pointer to InferenceState, hence the
 * InferenceState will be persisting even tho the lambda in streaming might go
 * out of scope and the handler already moved on
 */
std::shared_ptr<InferenceState> CreateInferenceState(LlamaServerContext& l) {
  return std::make_shared<InferenceState>(l);
}

Json::Value CreateEmbeddingPayload(const std::vector<float>& embedding,
                                   int prompt_tokens) {
  Json::Value dataItem;

  dataItem["object"] = "embedding";

  Json::Value embeddingArray(Json::arrayValue);
  for (const auto& value : embedding) {
    embeddingArray.append(value);
  }
  dataItem["embedding"] = embeddingArray;
  dataItem["index"] = 0;

  return dataItem;
}

Json::Value CreateFullReturnJson(const std::string& id,
                                 const std::string& model,
                                 const std::string& content,
                                 const std::string& system_fingerprint,
                                 int prompt_tokens, int completion_tokens,
                                 Json::Value finish_reason = Json::Value()) {
  Json::Value root;

  root["id"] = id;
  root["model"] = model;
  root["created"] = static_cast<int>(std::time(nullptr));
  root["object"] = "chat.completion";
  root["system_fingerprint"] = system_fingerprint;

  Json::Value choicesArray(Json::arrayValue);
  Json::Value choice;

  choice["index"] = 0;
  Json::Value message;
  message["role"] = "assistant";
  message["content"] = content;
  choice["message"] = message;
  choice["finish_reason"] = finish_reason;

  choicesArray.append(choice);
  root["choices"] = choicesArray;

  Json::Value usage;
  usage["prompt_tokens"] = prompt_tokens;
  usage["completion_tokens"] = completion_tokens;
  usage["total_tokens"] = prompt_tokens + completion_tokens;
  root["usage"] = usage;

  return root;
}

std::string CreateReturnJson(const std::string& id, const std::string& model,
                             const std::string& content,
                             Json::Value finish_reason = Json::Value()) {
  Json::Value root;

  root["id"] = id;
  root["model"] = model;
  root["created"] = static_cast<int>(std::time(nullptr));
  root["object"] = "chat.completion.chunk";

  Json::Value choicesArray(Json::arrayValue);
  Json::Value choice;

  choice["index"] = 0;
  Json::Value delta;
  delta["content"] = content;
  choice["delta"] = delta;
  choice["finish_reason"] = finish_reason;

  choicesArray.append(choice);
  root["choices"] = choicesArray;

  Json::StreamWriterBuilder writer;
  writer["indentation"] = "";  // This sets the indentation to an empty string,
                               // producing compact output.
  return Json::writeString(writer, root);
}
}  // namespace

// derepted this function because we no longer support change log when load model
void LlamaEngine::SetLoggerOption(const Json::Value& json_body) {
  // if (!json_body["log_option"].isNull()) {
  //   int log_option = json_body["log_option"].asInt();
  //   if (log_option != kFileLoggerOption) {
  //     // Revert to default trantor logger output function
  //     trantor::Logger::setOutputFunction(
  //         [](const char* msg, const uint64_t len) {
  //           fwrite(msg, 1, static_cast<size_t>(len), stdout);
  //         },
  //         []() { fflush(stdout); });
  //   } else {
  //     std::string log_path =
  //         json_body.get("log_path", "./logs/cortex.log").asString();
  //     int max_log_lines = json_body.get("max_log_lines", 100000).asInt();
  //     trantor::FileLogger asyncFileLogger;
  //     asyncFileLogger.setFileName(log_path);
  //     asyncFileLogger.setMaxLines(max_log_lines);  // Keep last 100000 lines
  //     // asyncFileLogger.startLogging();
  //     trantor::Logger::setOutputFunction(
  //         [&](const char* msg, const uint64_t len) {
  //           asynce_file_logger_->output_(msg, len);
  //         },
  //         [&]() { asynce_file_logger_->flush(); });
  //   }
  // } else {
  //   // For backward compatible
  //   trantor::Logger::setOutputFunction(
  //       [](const char* msg, const uint64_t len) {
  //         fwrite(msg, 1, static_cast<size_t>(len), stdout);
  //       },
  //       []() { fflush(stdout); });
  // }

  // if (!json_body["log_level"].isNull()) {
  //   std::string log_level = json_body["log_level"].asString();
  //   if (log_level == "trace") {
  //     trantor::Logger::setLogLevel(trantor::Logger::kTrace);
  //   } else if (log_level == "debug") {
  //     trantor::Logger::setLogLevel(trantor::Logger::kDebug);
  //   } else if (log_level == "info") {
  //     trantor::Logger::setLogLevel(trantor::Logger::kInfo);
  //   } else if (log_level == "warn") {
  //     trantor::Logger::setLogLevel(trantor::Logger::kWarn);
  //   } else if (log_level == "fatal") {
  //     trantor::Logger::setLogLevel(trantor::Logger::kFatal);
  //   } else {
  //     trantor::Logger::setLogLevel(trantor::Logger::kError);
  //   }
  // } else {
  //   trantor::Logger::setLogLevel(trantor::Logger::kDebug);
  // }
}

LlamaEngine::LlamaEngine(int log_option) {
  trantor::Logger::setLogLevel(trantor::Logger::kInfo);
  if (log_option == kFileLoggerOption) {
    asynce_file_logger_ = std::make_unique<trantor::FileLogger>();
  }

  log_disable();

  llama_log_set(
      [](ggml_log_level level, const char* text, void* user_data) {
        (void)level;
        (void)user_data;
        if (level == GGML_LOG_LEVEL_ERROR) {
          LOG_ERROR << text;
        } else if (level == GGML_LOG_LEVEL_DEBUG) {
          LOG_DEBUG << text;
        } else if (level == GGML_LOG_LEVEL_WARN) {
          LOG_WARN << text;
        } else {
          LOG_INFO << text;
        }
      },
      nullptr);
}

LlamaEngine::~LlamaEngine() {
  for (auto& [_, si] : server_map_) {
    auto& l = si.ctx;
    l.ReleaseResources();
  }
  server_map_.clear();
  asynce_file_logger_.reset();
}

void LlamaEngine::HandleChatCompletion(
    std::shared_ptr<Json::Value> json_body,
    std::function<void(Json::Value&&, Json::Value&&)>&& callback) {
  // Check if model is loaded
  if (CheckModelLoaded(callback, llama_utils::GetModelId(*json_body))) {
    // Model is loaded
    // Do Inference
    HandleInferenceImpl(llama::inferences::fromJson(json_body),
                        std::move(callback));
  }
}

void LlamaEngine::HandleEmbedding(
    std::shared_ptr<Json::Value> json_body,
    std::function<void(Json::Value&&, Json::Value&&)>&& callback) {
  // Check if model is loaded
  if (CheckModelLoaded(callback, llama_utils::GetModelId(*json_body))) {
    // Run embedding
    HandleEmbeddingImpl(json_body, std::move(callback));
  }
}

void LlamaEngine::LoadModel(
    std::shared_ptr<Json::Value> json_body,
    std::function<void(Json::Value&&, Json::Value&&)>&& callback) {
  // SetLoggerOption(*json_body); // dont update log option when load model
  if (std::exchange(print_version_, false)) {
#if defined(CORTEXLLAMA_VERSION)
    LOG_INFO << "cortex.llamacpp version: " << CORTEXLLAMA_VERSION;
#else
    LOG_INFO << "cortex.llamacpp version: default_version";
#endif
  }
  auto model_id = llama_utils::GetModelId(*json_body);
  if (model_id.empty()) {
    LOG_INFO << "Model id is empty in request";
    Json::Value jsonResp;
    jsonResp["message"] = "No model id found in request body";
    Json::Value status;
    status["is_done"] = false;
    status["has_error"] = true;
    status["is_stream"] = false;
    status["status_code"] = k400BadRequest;
    callback(std::move(status), std::move(jsonResp));
    return;
  }

  if (auto si = server_map_.find(model_id);
      si != server_map_.end() && si->second.ctx.model_loaded_external) {
    LOG_INFO << "Model already loaded";
    Json::Value jsonResp;
    jsonResp["message"] = "Model already loaded";
    Json::Value status;
    status["is_done"] = true;
    status["has_error"] = false;
    status["is_stream"] = false;
    status["status_code"] = k409Conflict;
    callback(std::move(status), std::move(jsonResp));
    return;
  }

  if (!LoadModelImpl(json_body)) {
    // Error occurred during model loading
    Json::Value jsonResp;
    jsonResp["message"] = "Failed to load model";
    Json::Value status;
    status["is_done"] = false;
    status["has_error"] = true;
    status["is_stream"] = false;
    status["status_code"] = k500InternalServerError;
    callback(std::move(status), std::move(jsonResp));
  } else {
    // Model loaded successfully
    Json::Value jsonResp;
    jsonResp["message"] = "Model loaded successfully";
    Json::Value status;
    status["is_done"] = true;
    status["has_error"] = false;
    status["is_stream"] = false;
    status["status_code"] = k200OK;
    callback(std::move(status), std::move(jsonResp));
    LOG_INFO << "Model loaded successfully: " << model_id;
  }
}

void LlamaEngine::UnloadModel(
    std::shared_ptr<Json::Value> json_body,
    std::function<void(Json::Value&&, Json::Value&&)>&& callback) {
  auto model_id = llama_utils::GetModelId(*json_body);
  if (CheckModelLoaded(callback, model_id)) {
    auto& l = server_map_[model_id].ctx;
    l.ReleaseResources();

    Json::Value jsonResp;
    jsonResp["message"] = "Model unloaded successfully";
    Json::Value status;
    status["is_done"] = true;
    status["has_error"] = false;
    status["is_stream"] = false;
    status["status_code"] = k200OK;
    callback(std::move(status), std::move(jsonResp));

    server_map_.erase(model_id);
    LOG_INFO << "Model unloaded successfully";
  }
}

void LlamaEngine::GetModelStatus(
    std::shared_ptr<Json::Value> json_body,
    std::function<void(Json::Value&&, Json::Value&&)>&& callback) {

  auto model_id = llama_utils::GetModelId(*json_body);
  if (auto is_loaded = CheckModelLoaded(callback, model_id); is_loaded) {
    // CheckModelLoaded gurantees that model_id exists in server_ctx_map;
    auto si = server_map_.find(model_id);
    Json::Value jsonResp;
    jsonResp["model_loaded"] = is_loaded;
    jsonResp["model_data"] = si->second.ctx.GetModelProps().dump();
    Json::Value status;
    status["is_done"] = true;
    status["has_error"] = false;
    status["is_stream"] = false;
    status["status_code"] = k200OK;
    callback(std::move(status), std::move(jsonResp));
    LOG_INFO << "Model status responded";
  }
}

void LlamaEngine::GetModels(
    std::shared_ptr<Json::Value> json_body,
    std::function<void(Json::Value&&, Json::Value&&)>&& callback) {
  Json::Value json_resp;
  Json::Value model_array(Json::arrayValue);
  for (const auto& [m, s] : server_map_) {
    if (s.ctx.model_loaded_external) {
      Json::Value val;
      val["id"] = m;
      val["engine"] = "cortex.llamacpp";
      val["start_time"] = s.start_time;
      val["vram"] = "-";
      val["ram"] = "-";
      val["object"] = "model";
      model_array.append(val);
    }
  }

  json_resp["object"] = "list";
  json_resp["data"] = model_array;

  Json::Value status;
  status["is_done"] = true;
  status["has_error"] = false;
  status["is_stream"] = false;
  status["status_code"] = k200OK;
  callback(std::move(status), std::move(json_resp));
  LOG_INFO << "Running models responded";
}
// should decrepted this function because it no longer used in cortex cpp
void LlamaEngine::SetFileLogger(int max_log_lines,
                                const std::string& log_path) {
  if (!asynce_file_logger_) {
    asynce_file_logger_ = std::make_unique<trantor::FileLogger>();
  }
  asynce_file_logger_->setFileName(log_path);
  asynce_file_logger_->setMaxLines(max_log_lines);  // Keep last 100000 lines
  asynce_file_logger_->startLogging();
  trantor::Logger::setOutputFunction(
      [&](const char* msg, const uint64_t len) {
        asynce_file_logger_->output_(msg, len);
      },
      [&]() { asynce_file_logger_->flush(); });
  llama_log_set(
      [](ggml_log_level level, const char* text, void* user_data) {
        (void)level;
        (void)user_data;
        if (level == GGML_LOG_LEVEL_ERROR) {
          LOG_ERROR << text;
        } else if (level == GGML_LOG_LEVEL_DEBUG) {
          LOG_DEBUG << text;
        } else if (level == GGML_LOG_LEVEL_WARN) {
          LOG_WARN << text;
        } else {
          LOG_INFO << text;
        }
      },
      nullptr);
}

bool LlamaEngine::LoadModelImpl(std::shared_ptr<Json::Value> json_body) {
  gpt_params params;
  std::string model_type;
  auto model_id = llama_utils::GetModelId(*json_body);
  // By default will setting based on number of handlers
  if (json_body) {
    if (!json_body->operator[]("mmproj").isNull()) {
      LOG_INFO << "MMPROJ FILE detected, multi-model enabled!";
      params.mmproj = json_body->operator[]("mmproj").asString();
    }
    if (!json_body->operator[]("grp_attn_n").isNull()) {
      params.grp_attn_n = json_body->operator[]("grp_attn_n").asInt();
    }
    if (!json_body->operator[]("grp_attn_w").isNull()) {
      params.grp_attn_w = json_body->operator[]("grp_attn_w").asInt();
    }
    if (!json_body->operator[]("mlock").isNull()) {
      params.use_mlock = json_body->operator[]("mlock").asBool();
    }

    if (!json_body->operator[]("grammar_file").isNull()) {
      std::string grammar_file =
          json_body->operator[]("grammar_file").asString();
      std::ifstream file(grammar_file);
      if (!file) {
        LOG_ERROR << "Grammar file not found";
        return false;
      } else {
        std::stringstream grammarBuf;
        grammarBuf << file.rdbuf();
        server_map_[model_id].grammar_file_content = grammarBuf.str();
      }
    };

    Json::Value model_path_v0 = json_body->operator[]("llama_model_path");
    Json::Value model_path_v1 = json_body->operator[]("model_path");
    auto model_path = model_path_v0.isNull() ? model_path_v1 : model_path_v0;
    if (model_path.isNull()) {
      LOG_ERROR << "Missing model path in request";
      return false;
    } else {
      if (std::filesystem::exists(
              std::filesystem::path(model_path.asString()))) {
        params.model = model_path.asString();
      } else {
        LOG_ERROR << "Could not find model in path " << model_path.asString();
      }
    }

    params.n_gpu_layers =
        json_body->get("ngl", 300)
            .asInt();  // change from 100 -> 300 since llama 3.1 has 292 gpu layers
    params.n_ctx = json_body->get("ctx_len", 2048).asInt();
    model_type = json_body->get("model_type", "llm").asString();
    // In case of embedding only model, we set default = true
    params.embedding =
        json_body->get("embedding", model_type == "embedding").asBool();
    params.n_batch = json_body->get("n_batch", 2048).asInt();
    params.n_ubatch = json_body->get("n_ubatch", params.n_batch).asInt();
    // Check if n_parallel exists in json_body, if not, set to drogon_thread
    params.n_parallel = json_body->get("n_parallel", 1).asInt();
    LOG_INFO << "Number of parallel is set to " << params.n_parallel;
    params.cpuparams.n_threads =
        json_body->get("cpu_threads", std::thread::hardware_concurrency())
            .asInt();
    params.cont_batching =
        json_body->get("cont_batching", true)
            .asBool();  // default true according to llama.cpp upstream

    params.cache_type_k = json_body->get("cache_type", kTypeF16).asString();
    if (!IsValidCacheType(params.cache_type_k)) {
      LOG_WARN << "Unsupported cache type: " << params.cache_type_k
               << ", fallback to f16";
      params.cache_type_k = kTypeF16;
    }
    params.cache_type_v = params.cache_type_k;
    LOG_DEBUG << "cache_type: " << params.cache_type_k;

    auto fa = json_body->get("flash_attn", true).asBool();
    auto force_enable_fa = params.cache_type_k != kTypeF16;
    if (force_enable_fa) {
      LOG_DEBUG << "Using KV cache quantization, force enable Flash Attention";
    }
    params.flash_attn = fa || force_enable_fa;
    if (params.flash_attn) {
      LOG_DEBUG << "Enabled Flash Attention";
    }

    params.use_mmap = json_body->get("use_mmap", true).asBool();
    if (!params.use_mmap) {
      LOG_DEBUG << "Disabled mmap";
    }
    params.n_predict = json_body->get("n_predict", -1).asInt();
    params.prompt = json_body->get("prompt", "").asString();
    params.conversation = json_body->get("conversation", false).asBool();
    params.special = json_body->get("special", false).asBool();

    server_map_[model_id].caching_enabled =
        json_body->get("caching_enabled", true).asBool();
    server_map_[model_id].user_prompt =
        json_body->get("user_prompt", "USER: ").asString();
    server_map_[model_id].ai_prompt =
        json_body->get("ai_prompt", "ASSISTANT: ").asString();
    server_map_[model_id].system_prompt =
        json_body->get("system_prompt", "ASSISTANT's RULE: ").asString();
    server_map_[model_id].pre_prompt =
        json_body->get("pre_prompt", "").asString();
    server_map_[model_id].repeat_last_n =
        json_body->get("repeat_last_n", 32).asInt();
    server_map_[model_id].stop_words = (*json_body)["stop"];
    LOG_DEBUG << "stop: " << server_map_[model_id].stop_words.toStyledString();

    if (!json_body->operator[]("llama_log_folder").isNull()) {
      log_enable();
      std::string llama_log_folder =
          json_body->operator[]("llama_log_folder").asString();
      log_set_target(llama_log_folder + "llama.log");
    }  // Set folder for llama log
  }
  if (params.model_alias == "unknown") {
    params.model_alias = params.model;
  }

  if (ShouldInitBackend()) {
    llama_backend_init();

    // LOG_INFO_LLAMA("build info",
    //                {{"build", BUILD_NUMBER}, {"commit", BUILD_COMMIT}});

    // The log below will output to terminal automatically, we need output to be configurable
    // LOG_INFO_LLAMA("system info",
    //                {
    //                    {"n_threads", params.n_threads},
    //                    {"total_threads", std::thread::hardware_concurrency()},
    //                    {"system_info", llama_print_system_info()},
    //                });
    LOG_INFO << "system info: " << "{'n_thread': " << params.cpuparams.n_threads
             << ", 'total_threads': " << std::thread::hardware_concurrency()
             << ". 'system_info': '" << llama_print_system_info() << "'}";
  }

  // load the model
  if (!server_map_[model_id].ctx.LoadModel(params)) {
    LOG_ERROR << "Error loading the model";
    // TODO use ScopeExit
    server_map_.erase(model_id);
    return false;  // Indicate failure
  }

  if (model_type == "llm") {
    server_map_[model_id].ctx.model_type = ModelType::kLlm;
  } else {
    server_map_[model_id].ctx.model_type = ModelType::kEmbedding;
  }
  server_map_[model_id].ctx.Initialize();

  server_map_[model_id].q = std::make_unique<trantor::ConcurrentTaskQueue>(
      params.n_parallel, model_id);
  server_map_[model_id].start_time =
      std::chrono::system_clock::now().time_since_epoch() /
      std::chrono::milliseconds(1);

  // For model like nomic-embed-text-v1.5.f16.gguf, etc, we don't need to warm up model.
  // So we use this variable to differentiate with other models
  if (server_map_[model_id].ctx.model_type == ModelType::kLlm) {
    WarmUpModel(model_id);
  }
  return true;
}

void LlamaEngine::HandleInferenceImpl(
    llama::inferences::ChatCompletionRequest&& completion,
    std::function<void(Json::Value&&, Json::Value&&)>&& callback) {
  assert(server_map_.find(completion.model_id) != server_map_.end());
  auto& si = server_map_[completion.model_id];
  if (si.ctx.model_type == ModelType::kEmbedding) {
    LOG_WARN << "Not support completion for embedding model";
    Json::Value jsonResp;
    jsonResp["message"] = "Not support completion for embedding model";
    Json::Value status;
    status["is_done"] = true;
    status["has_error"] = true;
    status["is_stream"] = false;
    status["status_code"] = k400BadRequest;
    callback(std::move(status), std::move(jsonResp));
    return;
  }
  std::string formatted_output = si.pre_prompt;
  int request_id = ++no_of_requests_;
  LOG_INFO << "Request " << request_id << ", " << "model "
           << completion.model_id << ": "
           << "Generating response for inference request";

  json data;
  json stopWords;
  int no_images = 0;
  // To set default value

  // Default values to enable auto caching
  data["cache_prompt"] = si.caching_enabled;
  data["n_keep"] = 0;

  // Passing load value
  data["repeat_last_n"] = si.repeat_last_n;
  auto stop_words_json =
      !completion.stop.empty() ? completion.stop : si.stop_words;
  LOG_INFO << "Request " << request_id << ": "
           << "Stop words:" << stop_words_json.toStyledString();

  data["stream"] = completion.stream;
  data["n_predict"] = completion.max_tokens;
  data["top_p"] = completion.top_p;
  data["temperature"] = completion.temperature;
  data["frequency_penalty"] = completion.frequency_penalty;
  data["presence_penalty"] = completion.presence_penalty;
  data["seed"] = completion.seed;
  data["dynatemp_range"] = completion.dynatemp_range;
  data["dynatemp_exponent"] = completion.dynatemp_exponent;
  data["top_k"] = completion.top_k;
  data["min_p"] = completion.min_p;
  data["tfs_z"] = completion.tfs_z;
  data["typical_p"] = completion.typ_p;
  data["repeat_last_n"] = completion.repeat_last_n;
  data["repeat_penalty"] = completion.penalty_repeat;
  data["mirostat"] = completion.mirostat;
  data["mirostat_tau"] = completion.mirostat_tau;
  data["mirostat_eta"] = completion.mirostat_eta;
  data["penalize_nl"] = completion.penalize_nl;
  data["ignore_eos"] = completion.ignore_eos;
  data["n_probs"] = completion.n_probs;
  data["min_keep"] = completion.min_keep;
  data["grammar"] = completion.grammar;
  int n_probs = completion.n_probs;
  const Json::Value& messages = completion.messages;

  if (!si.grammar_file_content.empty()) {
    data["grammar"] = si.grammar_file_content;
  };

  if (!si.ctx.multimodal) {
    for (const auto& message : messages) {
      std::string input_role = message["role"].asString();
      std::string role;
      if (input_role == "user") {
        role = si.user_prompt;
        std::string content = message["content"].asString();
        formatted_output += role + content;
      } else if (input_role == "assistant") {
        role = si.ai_prompt;
        std::string content = message["content"].asString();
        formatted_output += role + content;
      } else if (input_role == "system") {
        role = si.system_prompt;
        std::string content = message["content"].asString();
        formatted_output = role + content + formatted_output;

      } else {
        role = input_role;
        std::string content = message["content"].asString();
        formatted_output += role + content;
      }
    }
    formatted_output += si.ai_prompt;
  } else {
    data["image_data"] = json::array();
    for (const auto& message : messages) {
      std::string input_role = message["role"].asString();
      std::string role;
      if (input_role == "user") {
        formatted_output += role;
        for (auto content_piece : message["content"]) {
          role = si.user_prompt;

          json content_piece_image_data;
          content_piece_image_data["data"] = "";

          auto content_piece_type = content_piece["type"].asString();
          if (content_piece_type == "text") {
            auto text = content_piece["text"].asString();
            formatted_output += text;
          } else if (content_piece_type == "image_url") {
            auto image_url = content_piece["image_url"]["url"].asString();
            std::string base64_image_data;
            if (image_url.find("http") != std::string::npos) {
              LOG_INFO << "Request " << request_id << ": "
                       << "Remote image detected but not supported yet";
            } else if (image_url.find("data:image") != std::string::npos) {
              LOG_INFO << "Request " << request_id << ": "
                       << "Base64 image detected";
              base64_image_data = llama_utils::extractBase64(image_url);
              // LOG_INFO << "Request " << request_id << ": " << base64_image_data;
            } else {
              LOG_INFO << "Request " << request_id << ": "
                       << "Local image detected";
              llama_utils::processLocalImage(
                  image_url, [&](const std::string& base64Image) {
                    base64_image_data = base64Image;
                  });
              // LOG_INFO << "Request " << request_id << ": " << base64_image_data;
            }
            content_piece_image_data["data"] = base64_image_data;

            formatted_output += "[img-" + std::to_string(no_images) + "]";
            content_piece_image_data["id"] = no_images;
            data["image_data"].push_back(content_piece_image_data);
            no_images++;
          }
        }

      } else if (input_role == "assistant") {
        role = si.ai_prompt;
        std::string content = message["content"].asString();
        formatted_output += role + content;
      } else if (input_role == "system") {
        role = si.system_prompt;
        std::string content = message["content"].asString();
        formatted_output = role + content + formatted_output;

      } else {
        role = input_role;
        std::string content = message["content"].asString();
        formatted_output += role + content;
      }
    }
    formatted_output += si.ai_prompt;
    // LOG_INFO << "Request " << request_id << ": " << formatted_output;
  }

  data["prompt"] = formatted_output;
  for (const auto& sw : stop_words_json) {
    stopWords.push_back(sw.asString());
  }
  // specify default stop words
  // Ensure success case for chatML
  stopWords.push_back("<|im_end|>");
  stopWords.push_back(llama_utils::rtrim(si.user_prompt));
  data["stop"] = stopWords;

  bool is_streamed = data["stream"];
// Enable full message debugging
#ifdef DEBUG
  LOG_INFO << "Request " << request_id << ": " << "Current completion text";
  LOG_INFO << "Request " << request_id << ": " << formatted_output;
#endif

  if (is_streamed) {
    LOG_INFO << "Request " << request_id << ": "
             << "Streamed, waiting for respone";
    auto state = CreateInferenceState(si.ctx);

    // Queued task
    si.q->runTaskInQueue([cb = std::move(callback), state, data, request_id, n_probs]() {
      state->task_id = state->llama.RequestCompletion(data, false, false, -1);
      while (state->llama.model_loaded_external) {
        TaskResult result = state->llama.NextResult(state->task_id);
        if (!result.error) {
          std::string to_send;
          if (n_probs > 0){
            to_send = result.result_json["completion_probabilities"].dump();
          }else{
            to_send = result.result_json["content"];
          }
          // trim the leading space if it is the first token
          if (std::exchange(state->is_first_token, false)) {
            llama_utils::ltrim(to_send);
          }

          const std::string str =
              "data: " +
              CreateReturnJson(llama_utils::generate_random_string(20), "_",
                               to_send) +
              "\n\n";
          Json::Value respData;
          respData["data"] = str;
          Json::Value status;
          status["is_done"] = false;
          status["has_error"] = false;
          status["is_stream"] = true;
          status["status_code"] = k200OK;
          cb(std::move(status), std::move(respData));

          if (result.stop) {
            LOG_INFO << "Request " << request_id << ": " << "End of result";
            state->llama.RequestCancel(state->task_id);
            Json::Value respData;
            const std::string str =
                "data: " +
                CreateReturnJson(llama_utils::generate_random_string(20), "_",
                                 "", "stop") +
                "\n\n" + "data: [DONE]" + "\n\n";
            respData["data"] = str;
            Json::Value status;
            status["is_done"] = true;
            status["has_error"] = false;
            status["is_stream"] = true;
            status["status_code"] = k200OK;
            cb(std::move(status), std::move(respData));
            break;
          }

        } else {
          state->llama.RequestCancel(state->task_id);
          LOG_ERROR << "Request " << request_id << ": "
                    << "Error during inference";
          Json::Value respData;
          respData["data"] = std::string();
          Json::Value status;
          status["is_done"] = false;
          status["has_error"] = true;
          status["is_stream"] = true;
          status["status_code"] = k200OK;
          cb(std::move(status), std::move(respData));
          break;
        }
      }
      LOG_INFO << "Request " << request_id << ": "
               << "Task completed, release it";
      // Request completed, release it
      if (!state->llama.model_loaded_external) {
        LOG_WARN << "Model unloaded during inference";
        Json::Value respData;
        respData["data"] = std::string();
        Json::Value status;
        status["is_done"] = false;
        status["has_error"] = true;
        status["is_stream"] = true;
        status["status_code"] = k200OK;
        cb(std::move(status), std::move(respData));
      }
      LOG_INFO << "Request " << request_id << ": " << "Inference completed";
    });
  } else {
    auto state = CreateInferenceState(si.ctx);
    si.q->runTaskInQueue([this, request_id, state, cb = std::move(callback),
                          d = std::move(data)]() {
      Json::Value respData;
      int task_id = state->llama.RequestCompletion(d, false, false, -1);
      LOG_INFO << "Request " << request_id << ": "
               << "Non stream, waiting for respone";
      if (!json_value(d, "stream", false)) {
        bool has_error = false;
        std::string completion_text;
        TaskResult result = state->llama.NextResult(task_id);
        if (!result.error && result.stop) {
          int prompt_tokens = result.result_json["tokens_evaluated"];
          int predicted_tokens = result.result_json["tokens_predicted"];
          std::string to_send = result.result_json["content"];
          llama_utils::ltrim(to_send);
          respData = CreateFullReturnJson(
              llama_utils::generate_random_string(20), "_", to_send, "_",
              prompt_tokens, predicted_tokens);
        } else {
          bool has_error = true;
          respData["message"] = "Internal error during inference";
          LOG_ERROR << "Request " << request_id << ": "
                    << "Error during inference";
        }
        Json::Value status;
        status["is_done"] = true;
        status["has_error"] = has_error;
        status["is_stream"] = false;
        status["status_code"] = k200OK;
        cb(std::move(status), std::move(respData));

        LOG_INFO << "Request " << request_id << ": " << "Inference completed";
      }
    });
  }
}

void LlamaEngine::HandleEmbeddingImpl(
    std::shared_ptr<Json::Value> json_body,
    std::function<void(Json::Value&&, Json::Value&&)>&& callback) {
  auto model_id = llama_utils::GetModelId(*json_body);
  assert(server_map_.find(model_id) != server_map_.end());
  int request_id = ++no_of_requests_;
  LOG_INFO << "Request " << request_id << ", " << "model " << model_id << ": "
           << "Generating response for embedding request";
  // Queue embedding task
  auto state = CreateInferenceState(server_map_[model_id].ctx);

  server_map_[model_id].q->runTaskInQueue([this, state, json_body, callback,
                                           request_id,
                                           mid = std::move(model_id)]() {
    Json::Value responseData(Json::arrayValue);

    int prompt_tokens = 0;
    if (json_body->isMember("input")) {
      const Json::Value& input = (*json_body)["input"];
      if (input.isString()) {
        // Process the single string input
        state->task_id = state->llama.RequestCompletion(
            {{"prompt", input.asString()}, {"n_predict", 0}}, false, true, -1);
        TaskResult result = state->llama.NextResult(state->task_id);
        prompt_tokens +=
            static_cast<int>(result.result_json["tokens_evaluated"]);
        std::vector<float> embedding_result = result.result_json["embedding"];
        responseData.append(CreateEmbeddingPayload(embedding_result, 0));
      } else if (input.isArray()) {
        // Process each element in the array input
        for (const auto& elem : input) {
          if (elem.isString()) {
            const int task_id = state->llama.RequestCompletion(
                {{"prompt", elem.asString()}, {"n_predict", 0}}, false, true,
                -1);
            TaskResult result = state->llama.NextResult(task_id);
            int cur_pt = result.result_json["tokens_evaluated"];
            prompt_tokens += cur_pt;
            std::vector<float> embedding_result =
                result.result_json["embedding"];
            responseData.append(
                CreateEmbeddingPayload(embedding_result, cur_pt));
          }
        }
      }
    }

    Json::Value root;
    root["data"] = responseData;
    root["model"] = mid;
    root["object"] = "list";
    Json::Value usage;
    usage["prompt_tokens"] = prompt_tokens;
    usage["total_tokens"] = prompt_tokens;
    root["usage"] = usage;
    Json::Value status;
    status["is_done"] = true;
    status["has_error"] = false;
    status["is_stream"] = false;
    status["status_code"] = k200OK;
    callback(std::move(status), std::move(root));

    LOG_INFO << "Request " << request_id << ": " << "Embedding completed";
  });
}

bool LlamaEngine::CheckModelLoaded(
    std::function<void(Json::Value&&, Json::Value&&)>& callback,
    const std::string& model_id) {
  if (auto si = server_map_.find(model_id);
      si == server_map_.end() || !si->second.ctx.model_loaded_external) {
    LOG_WARN << "Error: model_id: " << model_id
             << ", existed: " << (si != server_map_.end())
             << ", loaded: " << false;
    Json::Value jsonResp;
    jsonResp["message"] =
        "Model has not been loaded, please load model into cortex.llamacpp";
    Json::Value status;
    status["is_done"] = false;
    status["has_error"] = true;
    status["is_stream"] = false;
    status["status_code"] = k409Conflict;
    callback(std::move(status), std::move(jsonResp));
    return false;
  }
  return true;
}

void LlamaEngine::WarmUpModel(const std::string& model_id) {
  if (auto si = server_map_.find(model_id); si != server_map_.end()) {
    json pseudo;

    LOG_INFO << "Warm-up model: " << model_id;
    pseudo["prompt"] = "Hello";
    pseudo["n_predict"] = 2;
    pseudo["stream"] = false;
    const int task_id =
        si->second.ctx.RequestCompletion(pseudo, false, false, -1);
    TaskResult result = si->second.ctx.NextResult(task_id);
    if (!result.error && result.stop) {
      LOG_INFO << result.result_json.dump(-1, ' ', false,
                                          json::error_handler_t::replace);
    }
  } else {
    LOG_WARN << "Model not found " << model_id;
  }
}

bool LlamaEngine::ShouldInitBackend() const {
  // May have race condition here, need to check
  for (auto& [_, l] : server_map_) {
    if (l.ctx.model_loaded_external)
      return false;
  }
  return true;
}

extern "C" {
EngineI* get_engine() {
  return new LlamaEngine();
}
}