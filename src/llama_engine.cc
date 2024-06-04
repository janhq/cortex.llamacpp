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

LlamaEngine::LlamaEngine() {
  log_disable();
}

LlamaEngine::~LlamaEngine() {}

void LlamaEngine::HandleChatCompletion(
    std::shared_ptr<Json::Value> jsonBody,
    std::function<void(Json::Value&&, Json::Value&&)>&& callback) {
  // Check if model is loaded
  if (CheckModelLoaded(callback, llama_utils::GetModelId(*jsonBody))) {
    // Model is loaded
    // Do Inference
    HandleInferenceImpl(llama::inferences::fromJson(jsonBody),
                        std::move(callback));
  }
}

void LlamaEngine::HandleEmbedding(
    std::shared_ptr<Json::Value> jsonBody,
    std::function<void(Json::Value&&, Json::Value&&)>&& callback) {
  // Check if model is loaded
  if (CheckModelLoaded(callback, llama_utils::GetModelId(*jsonBody))) {
    // Run embedding
    HandleEmbeddingImpl(jsonBody, std::move(callback));
  }
}

void LlamaEngine::LoadModel(
    std::shared_ptr<Json::Value> jsonBody,
    std::function<void(Json::Value&&, Json::Value&&)>&& callback) {
  if (!llama_utils::isAVX2Supported() && ggml_cpu_has_avx2()) {
    LOG_ERROR << "AVX2 is not supported by your processor";
    Json::Value jsonResp;
    jsonResp["message"] =
        "AVX2 is not supported by your processor, please download and replace "
        "the correct Nitro asset version";
    Json::Value status;
    status["is_done"] = false;
    status["has_error"] = true;
    status["is_stream"] = false;
    status["status_code"] = k500InternalServerError;
    callback(std::move(status), std::move(jsonResp));
    return;
  }

  auto model_id = llama_utils::GetModelId(*jsonBody);
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

  if (!LoadModelImpl(jsonBody)) {
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
    std::shared_ptr<Json::Value> jsonBody,
    std::function<void(Json::Value&&, Json::Value&&)>&& callback) {
  auto model_id = llama_utils::GetModelId(*jsonBody);
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
    std::shared_ptr<Json::Value> jsonBody,
    std::function<void(Json::Value&&, Json::Value&&)>&& callback) {

  auto model_id = llama_utils::GetModelId(*jsonBody);
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
    std::shared_ptr<Json::Value> jsonBody,
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

bool LlamaEngine::LoadModelImpl(std::shared_ptr<Json::Value> jsonBody) {
  gpt_params params;
  std::string model_type;
  auto model_id = llama_utils::GetModelId(*jsonBody);
  // By default will setting based on number of handlers
  if (jsonBody) {
    if (!jsonBody->operator[]("mmproj").isNull()) {
      LOG_INFO << "MMPROJ FILE detected, multi-model enabled!";
      params.mmproj = jsonBody->operator[]("mmproj").asString();
    }
    if (!jsonBody->operator[]("grp_attn_n").isNull()) {
      params.grp_attn_n = jsonBody->operator[]("grp_attn_n").asInt();
    }
    if (!jsonBody->operator[]("grp_attn_w").isNull()) {
      params.grp_attn_w = jsonBody->operator[]("grp_attn_w").asInt();
    }
    if (!jsonBody->operator[]("mlock").isNull()) {
      params.use_mlock = jsonBody->operator[]("mlock").asBool();
    }

    if (!jsonBody->operator[]("grammar_file").isNull()) {
      std::string grammar_file =
          jsonBody->operator[]("grammar_file").asString();
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

    Json::Value model_path = jsonBody->operator[]("llama_model_path");
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

    params.n_gpu_layers = jsonBody->get("ngl", 100).asInt();
    params.n_ctx = jsonBody->get("ctx_len", 2048).asInt();
    params.embedding = jsonBody->get("embedding", true).asBool();
    model_type = jsonBody->get("model_type", "llm").asString();
    params.n_batch = jsonBody->get("n_batch", 2048).asInt();
    params.n_ubatch = jsonBody->get("n_ubatch", params.n_batch).asInt();
    // Check if n_parallel exists in jsonBody, if not, set to drogon_thread
    params.n_parallel = jsonBody->get("n_parallel", 1).asInt();
    params.n_threads =
        jsonBody->get("cpu_threads", std::thread::hardware_concurrency())
            .asInt();
    params.cont_batching = jsonBody->get("cont_batching", false).asBool();

    // Check for backward compatible
    auto fa0 = jsonBody->get("flash-attn", false).asBool();
    auto fa1 = jsonBody->get("flash_attn", false).asBool();
    params.flash_attn = fa0 || fa1;
    if (params.flash_attn) {
      LOG_DEBUG << "Enabled Flash Attention";
    }
    server_map_[model_id].caching_enabled =
        jsonBody->get("caching_enabled", false).asBool();
    server_map_[model_id].user_prompt =
        jsonBody->get("user_prompt", "USER: ").asString();
    server_map_[model_id].ai_prompt =
        jsonBody->get("ai_prompt", "ASSISTANT: ").asString();
    server_map_[model_id].system_prompt =
        jsonBody->get("system_prompt", "ASSISTANT's RULE: ").asString();
    server_map_[model_id].pre_prompt =
        jsonBody->get("pre_prompt", "").asString();
    server_map_[model_id].repeat_last_n =
        jsonBody->get("repeat_last_n", 32).asInt();
    server_map_[model_id].stop_words = (*jsonBody)["stop"];
    LOG_DEBUG << "stop: " << server_map_[model_id].stop_words.toStyledString();

    if (!jsonBody->operator[]("llama_log_folder").isNull()) {
      log_enable();
      std::string llama_log_folder =
          jsonBody->operator[]("llama_log_folder").asString();
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
    LOG_INFO_LLAMA("system info",
                   {
                       {"n_threads", params.n_threads},
                       {"total_threads", std::thread::hardware_concurrency()},
                       {"system_info", llama_print_system_info()},
                   });
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
           << "Generating reponse for inference request";

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
              LOG_INFO << "Request " << request_id << ": " << base64_image_data;
            } else {
              LOG_INFO << "Request " << request_id << ": "
                       << "Local image detected";
              llama_utils::processLocalImage(
                  image_url, [&](const std::string& base64Image) {
                    base64_image_data = base64Image;
                  });
              LOG_INFO << "Request " << request_id << ": " << base64_image_data;
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
    LOG_INFO << "Request " << request_id << ": " << formatted_output;
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
    si.q->runTaskInQueue([cb = std::move(callback), state, data, request_id]() {
      state->task_id = state->llama.RequestCompletion(data, false, false, -1);
      while (state->llama.model_loaded_external) {
        TaskResult result = state->llama.NextResult(state->task_id);
        if (!result.error) {
          std::string to_send = result.result_json["content"];
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
    std::shared_ptr<Json::Value> jsonBody,
    std::function<void(Json::Value&&, Json::Value&&)>&& callback) {
  auto model_id = llama_utils::GetModelId(*jsonBody);
  assert(server_map_.find(model_id) != server_map_.end());
  int request_id = ++no_of_requests_;
  LOG_INFO << "Request " << request_id << ", " << "model " << model_id << ": "
           << "Generating reponse for embedding request";
  // Queue embedding task
  auto state = CreateInferenceState(server_map_[model_id].ctx);

  server_map_[model_id].q->runTaskInQueue([this, state, jsonBody, callback,
                                           request_id]() {
    Json::Value responseData(Json::arrayValue);

    if (jsonBody->isMember("input")) {
      const Json::Value& input = (*jsonBody)["input"];
      if (input.isString()) {
        // Process the single string input
        state->task_id = state->llama.RequestCompletion(
            {{"prompt", input.asString()}, {"n_predict", 0}}, false, true, -1);
        TaskResult result = state->llama.NextResult(state->task_id);
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
            std::vector<float> embedding_result =
                result.result_json["embedding"];
            responseData.append(CreateEmbeddingPayload(embedding_result, 0));
          }
        }
      }
    }

    Json::Value root;
    root["data"] = responseData;
    root["model"] = "_";
    root["object"] = "list";
    Json::Value usage;
    usage["prompt_tokens"] = 0;
    usage["total_tokens"] = 0;
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