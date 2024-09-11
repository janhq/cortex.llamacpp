#include "llama_server_context.h"
#include "sampling.h"

namespace {
const std::string base64_chars =
    "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    "abcdefghijklmnopqrstuvwxyz"
    "0123456789+/";

bool is_base64(uint8_t c) {
  return (isalnum(c) || (c == '+') || (c == '/'));
}

std::vector<uint8_t> base64_decode(const std::string& encoded_string) {
  int i = 0;
  int j = 0;
  int in_ = 0;

  int in_len = encoded_string.size();

  uint8_t char_array_4[4];
  uint8_t char_array_3[3];

  std::vector<uint8_t> ret;

  while (in_len-- && (encoded_string[in_] != '=') &&
         is_base64(encoded_string[in_])) {
    char_array_4[i++] = encoded_string[in_];
    in_++;
    if (i == 4) {
      for (i = 0; i < 4; i++) {
        char_array_4[i] = base64_chars.find(char_array_4[i]);
      }

      char_array_3[0] =
          ((char_array_4[0]) << 2) + ((char_array_4[1] & 0x30) >> 4);
      char_array_3[1] =
          ((char_array_4[1] & 0xf) << 4) + ((char_array_4[2] & 0x3c) >> 2);
      char_array_3[2] = ((char_array_4[2] & 0x3) << 6) + char_array_4[3];

      for (i = 0; (i < 3); i++) {
        ret.push_back(char_array_3[i]);
      }
      i = 0;
    }
  }

  if (i) {
    for (j = i; j < 4; j++) {
      char_array_4[j] = 0;
    }

    for (j = 0; j < 4; j++) {
      char_array_4[j] = base64_chars.find(char_array_4[j]);
    }

    char_array_3[0] =
        ((char_array_4[0]) << 2) + ((char_array_4[1] & 0x30) >> 4);
    char_array_3[1] =
        ((char_array_4[1] & 0xf) << 4) + ((char_array_4[2] & 0x3c) >> 2);
    char_array_3[2] = ((char_array_4[2] & 0x3) << 6) + char_array_4[3];

    for (j = 0; (j < i - 1); j++) {
      ret.push_back(char_array_3[j]);
    }
  }

  return ret;
}

size_t common_part(const std::vector<llama_token>& a,
                   const std::vector<llama_token>& b) {
  size_t i;
  for (i = 0; i < a.size() && i < b.size() && a[i] == b[i]; i++) {}
  return i;
}

bool ends_with(const std::string& str, const std::string& suffix) {
  return str.size() >= suffix.size() &&
         0 == str.compare(str.size() - suffix.size(), suffix.size(), suffix);
}

size_t find_partial_stop_string(const std::string& stop,
                                const std::string& text) {
  if (!text.empty() && !stop.empty()) {
    const char text_last_char = text.back();
    for (int64_t char_index = stop.size() - 1; char_index >= 0; char_index--) {
      if (stop[char_index] == text_last_char) {
        const std::string current_partial = stop.substr(0, char_index + 1);
        if (ends_with(text, current_partial)) {
          return text.size() - char_index - 1;
        }
      }
    }
  }
  return std::string::npos;
}

// format incomplete utf-8 multibyte character for output
inline std::string tokens_to_output_formatted_string(const llama_context* ctx,
                                                     const llama_token token) {
  std::string out = token == -1 ? "" : llama_token_to_piece(ctx, token);
  // if the size is 1 and first bit is 1, meaning it's a partial character
  //   (size > 1 meaning it's already a known token)
  if (out.size() == 1 && (out[0] & 0x80) == 0x80) {
    std::stringstream ss;
    ss << std::hex << (out[0] & 0xff);
    std::string res(ss.str());
    out = "byte: \\x" + res;
  }
  return out;
}

// convert a vector of CompletionTokenOutput to json
inline json probs_vector_to_json(
    const llama_context* ctx, const std::vector<CompletionTokenOutput>& probs) {
  json out = json::array();
  for (const auto& prob : probs) {
    json probs_for_token = json::array();
    for (const auto& p : prob.probs) {
      std::string tok_str = tokens_to_output_formatted_string(ctx, p.tok);
      probs_for_token.push_back(json{
          {"tok_str", tok_str},
          {"prob", p.prob},
      });
    }
    std::string tok_str = tokens_to_output_formatted_string(ctx, prob.tok);
    out.push_back(json{
        {"content", tok_str},
        {"probs", probs_for_token},
    });
  }
  return out;
}

bool IsLlava_1_6(const std::string& model) {
  if (model.find("llava-v1.6") != std::string::npos) {
    return true;
  }
  return false;
}

}  // namespace

LlamaServerContext::~LlamaServerContext() {
  if (ctx) {
    llama_free(ctx);
    ctx = nullptr;
  }
  if (model) {
    llama_free_model(model);
    model = nullptr;
  }
}

bool LlamaServerContext::LoadModel(const gpt_params& params_) {
  params = params_;
  if (!params.mmproj.empty()) {
    multimodal = true;
    LOG_DEBUG << "Multi Modal Mode Enabled";
    clp_ctx = clip_model_load(params.mmproj.c_str(), /*verbosity=*/1);
    if (clp_ctx == nullptr) {
      LOG_ERROR_LLAMA("unable to load clip model", {{"model", params.mmproj}});
      return false;
    }

    // https://github.com/ggerganov/llama.cpp/blob/master/examples/llava/README.md
    // note llava-1.6 needs more context than llava-1.5, at least 3000 is needed (just run it at -c 4096)
    if (params.n_ctx < 4096 && IsLlava_1_6(params.model)) {
      params.n_ctx = 4096;
      LOG_DEBUG << "Request " << params.n_ctx
                << " for context length for llava-1.6";
    } else if (params.n_ctx <
               2048) {  // request larger context for the image embedding
      params.n_ctx = 2048;
      LOG_DEBUG << "Request " << params.n_ctx
                << " for context length for the image embedding";
    }
  }

  auto res = llama_init_from_gpt_params(params);
  model = res.model;
  ctx = res.context;
  if (model == nullptr) {
    LOG_ERROR_LLAMA("llama.cpp unable to load model",
                    {{"model", params.model}});
    return false;
  }

  if (multimodal) {
    const int n_embd_clip = clip_n_mmproj_embd(clp_ctx);
    const int n_embd_llm = llama_n_embd(model);
    if (n_embd_clip != n_embd_llm) {
      LOG_DEBUG << __func__ << ": embedding dim of the multimodal projector ("
                << n_embd_clip
                << ") is not "
                   "equal to that of LLaMA ("
                << n_embd_llm
                << "). Make sure that you use the "
                   "correct mmproj file.";
      llama_free(ctx);
      llama_free_model(model);
      return false;
    }
  }

  if (ctx == nullptr) {
    LOG_ERROR_LLAMA("Unable to get llama.cpp context", {});
    return false;
  }
  n_ctx = llama_n_ctx(ctx);

  add_bos_token = llama_add_bos_token(model);
  has_eos_token = !llama_add_eos_token(model);

  return true;
}

void LlamaServerContext::Initialize() {
  id_gen = 0;

  // create slots
  all_slots_are_idle = true;

  const int32_t n_ctx_slot = n_ctx / params.n_parallel;

  LOG_DEBUG << "Available slots: ";
  for (int i = 0; i < params.n_parallel; i++) {
    LlamaClientSlot slot;

    slot.id = i;
    slot.n_ctx = n_ctx_slot;
    slot.Reset();

    LOG_DEBUG << " -> Slot " << slot.id << " - max context: " << n_ctx_slot;
    slots.push_back(slot);
  }

  try {
    batch = llama_batch_init(n_ctx, 0, params.n_parallel);
  } catch (const std::exception& e) {
    LOG_ERROR_LLAMA("Failed to allocate llama.cpp batch metadata",
                    {{"exception", e.what()},
                     {"n_tokens_alloc", n_ctx},
                     {"embd", 0},
                     {"n_seq_max", params.n_parallel}});
  }

  // empty system prompt
  system_prompt = "";
  system_tokens.clear();

  model_loaded_external = true;
  LOG_INFO << "Started background task here!";
  bgr_thread =
      std::thread(std::bind(&LlamaServerContext::DoBackgroundTasks, this));
}

void LlamaServerContext::KvCacheClear() {
  LOG_DEBUG << "Clear the entire KV cache";
  // clear the entire KV cache
  llama_kv_cache_clear(ctx);
  clean_kv_cache = false;
}

json LlamaServerContext::GetModelProps() {
  return GetFormatedGeneration(slots[0]);
}

int LlamaServerContext::RequestCompletion(json data, bool infill,
                                          bool embedding, int multitask_id) {
  // From this commit: 'llama : allow pooled embeddings on any model (#7477)'
  // we need to explicitly set embedding flad for each request
  llama_set_embeddings(ctx, embedding);

  TaskServer task;
  task.id = id_gen++;
  task.target_id = 0;
  task.data = std::move(data);
  task.infill_mode = infill;
  task.embedding_mode = embedding;
  task.type = TaskType::kCompletionTask;
  task.multitask_id = multitask_id;

  // when a completion task's prompt array is not a singleton, we split it
  // into multiple requests
  if (task.data.at("prompt").size() > 1) {
    return SplitMultipromptTask(task);
  }

  // otherwise, it's a single-prompt task, we actually queue it
  {
    std::lock_guard<std::mutex> lock(mutex_tasks);
    queue_tasks.push_back(task);
  }
  condition_tasks.notify_one();
  return task.id;
}

TaskResult LlamaServerContext::NextResult(int task_id) {
  while (true) {
    std::unique_lock<std::mutex> lock(mutex_results);
    condition_results.wait(lock, [&] { return !queue_results.empty(); });

    for (int i = 0; i < (int)queue_results.size(); i++) {
      // for now, tasks that have associated parent multitasks just get erased
      // once multitask picks up the result
      if (queue_results[i].multitask_id == task_id) {
        UpdateMultiTask(task_id, queue_results[i].id, queue_results[i]);
        queue_results.erase(queue_results.begin() + i);
        continue;
      }

      if (queue_results[i].id == task_id) {
        if (queue_results[i].multitask_id != -1) {
          LOG_ERROR_LLAMA("Incorrect multitask ID", {{"task_id", task_id}});
        }
        TaskResult res = queue_results[i];
        queue_results.erase(queue_results.begin() + i);
        return res;
      }
    }
  }

  // never reached
  // return TaskResult{-1, false, false, {}};
}

void LlamaServerContext::RequestCancel(int task_id) {
  TaskServer task;
  task.id = id_gen++;
  task.type = TaskType::kCancelTask;
  task.target_id = task_id;
  {
    std::lock_guard<std::mutex> lock(mutex_tasks);
    queue_tasks.push_back(task);
  }
  condition_tasks.notify_one();
}

void LlamaServerContext::ReleaseResources() {
  if (model_loaded_external) {
    LOG_INFO << "Releasing llama_server_context resources";
    model_loaded_external = false;
    condition_tasks.notify_one();

    if (bgr_thread.joinable()) {
      bgr_thread.join();
    }

    llama_free(ctx);
    llama_free_model(model);
    ctx = nullptr;
    model = nullptr;
    LOG_INFO << "Released llama_server_context resources";
  }
}

std::vector<llama_token> LlamaServerContext::Tokenize(const json& json_prompt,
                                                      bool add_bos) const {
  // TODO: currently, we tokenize using special tokens by default
  //       this is not always correct (see
  //       https://github.com/ggerganov/llama.cpp/pull/4160#issuecomment-1824826216)
  //       but it's better compared to completely ignoring ChatML and other
  //       chat templates
  const bool TMP_FORCE_SPECIAL = true;

  // If `add_bos` is true, we only add BOS, when json_prompt is a string,
  // or the first element of the json_prompt array is a string.
  std::vector<llama_token> prompt_tokens;

  if (json_prompt.is_array()) {
    bool first = true;
    for (const auto& p : json_prompt) {
      if (p.is_string()) {
        auto s = p.template get<std::string>();
        std::vector<llama_token> p;
        if (first) {
          p = ::llama_tokenize(ctx, s, add_bos, TMP_FORCE_SPECIAL);
          first = false;
        } else {
          p = ::llama_tokenize(ctx, s, false, TMP_FORCE_SPECIAL);
        }
        prompt_tokens.insert(prompt_tokens.end(), p.begin(), p.end());
      } else {
        if (first) {
          first = false;
        }
        prompt_tokens.push_back(p.template get<llama_token>());
      }
    }
  } else {
    auto s = json_prompt.template get<std::string>();
    prompt_tokens = ::llama_tokenize(ctx, s, add_bos, TMP_FORCE_SPECIAL);
  }

  return prompt_tokens;
}

LlamaClientSlot* LlamaServerContext::GetSlot(int id) {
  int64_t t_last = ggml_time_us();
  LlamaClientSlot* last_used = nullptr;

  for (LlamaClientSlot& slot : slots) {
    if (slot.id == id && slot.Available()) {
      return &slot;
    }

    if (slot.Available() && slot.t_last_used < t_last) {
      last_used = &slot;
      t_last = slot.t_last_used;
    }
  }

  return last_used;
}

bool LlamaServerContext::LaunchSlotWithData(LlamaClientSlot*& slot, json data) {
  SlotParams default_params;
  // Sampling parameter defaults are loaded from the global server context (but individual requests can still override them)
  auto default_sparams = params.sparams;

  if (data.count("__oaicompat") != 0) {
    slot->oaicompat = true;
    slot->oaicompat_model =
        json_value(data, "model", std::string(DEFAULT_OAICOMPAT_MODEL));
  } else {
    slot->oaicompat = false;
    slot->oaicompat_model = "";
  }

  slot->params.stream = json_value(data, "stream", false);
  slot->params.cache_prompt = json_value(data, "cache_prompt", false);
  slot->params.n_predict =
      json_value(data, "n_predict", default_params.n_predict);
  slot->sparams.top_k = json_value(data, "top_k", default_sparams.top_k);
  slot->sparams.top_p = json_value(data, "top_p", default_sparams.top_p);
  slot->sparams.min_p = json_value(data, "min_p", default_sparams.min_p);
  slot->sparams.tfs_z = json_value(data, "tfs_z", default_sparams.tfs_z);
  slot->sparams.typ_p = json_value(data, "typical_p", default_sparams.typ_p);
  slot->sparams.temp = json_value(data, "temperature", default_sparams.temp);
  slot->sparams.penalty_last_n =
      json_value(data, "repeat_last_n", default_sparams.penalty_last_n);
  slot->sparams.penalty_repeat =
      json_value(data, "repeat_penalty", default_sparams.penalty_repeat);
  slot->sparams.penalty_freq =
      json_value(data, "frequency_penalty", default_sparams.penalty_freq);
  slot->sparams.penalty_present =
      json_value(data, "presence_penalty", default_sparams.penalty_present);
  slot->sparams.mirostat =
      json_value(data, "mirostat", default_sparams.mirostat);
  slot->sparams.mirostat_tau =
      json_value(data, "mirostat_tau", default_sparams.mirostat_tau);
  slot->sparams.mirostat_eta =
      json_value(data, "mirostat_eta", default_sparams.mirostat_eta);
  slot->sparams.penalize_nl =
      json_value(data, "penalize_nl", default_sparams.penalize_nl);
  slot->params.n_keep = json_value(data, "n_keep", slot->params.n_keep);
  slot->params.seed = json_value(data, "seed", default_params.seed);
  slot->sparams.grammar = json_value(data, "grammar", default_sparams.grammar);
  slot->sparams.n_probs = json_value(data, "n_probs", default_sparams.n_probs);

  // infill
  if (data.count("input_prefix") != 0) {
    slot->params.input_prefix = data["input_prefix"];
  } else {
    slot->params.input_prefix = "";
  }

  if (data.count("input_suffix") != 0) {
    slot->params.input_suffix = data["input_suffix"];
  } else {
    slot->params.input_suffix = "";
  }

  if (data.count("prompt") != 0) {
    slot->prompt = data["prompt"];
  } else {
    slot->prompt = "";
  }

  {
    slot->sparams.logit_bias.clear();

    if (json_value(data, "ignore_eos", false) && has_eos_token) {
      slot->sparams.logit_bias.push_back({llama_token_eos(model), -INFINITY});
    }

    const auto& logit_bias = data.find("logit_bias");
    if (logit_bias != data.end() && logit_bias->is_array()) {
      const int n_vocab = llama_n_vocab(model);
      for (const auto& el : *logit_bias) {
        // TODO: we may want to throw errors here, in case "el" is incorrect
        if (el.is_array() && el.size() == 2) {
          float bias;
          if (el[1].is_number()) {
            bias = el[1].get<float>();
          } else if (el[1].is_boolean() && !el[1].get<bool>()) {
            bias = -INFINITY;
          } else {
            continue;
          }

          if (el[0].is_number_integer()) {
            llama_token tok = el[0].get<llama_token>();
            if (tok >= 0 && tok < n_vocab) {
              slot->sparams.logit_bias.push_back({tok, bias});
            }
          } else if (el[0].is_string()) {
            auto toks = llama_tokenize(model, el[0].get<std::string>(), false);
            for (auto tok : toks) {
              slot->sparams.logit_bias.push_back({tok, bias});
            }
          }
        }
      }
    }
  }

  slot->params.antiprompt.clear();

  const auto& stop = data.find("stop");
  if (stop != data.end() && stop->is_array()) {
    for (const auto& word : *stop) {
      if (!word.empty()) {
        slot->params.antiprompt.push_back(word);
      }
    }
  }

  if (multimodal) {
    const auto& images_data = data.find("image_data");
    if (images_data != data.end() && images_data->is_array()) {
      for (const auto& img : *images_data) {
        const std::vector<uint8_t> image_buffer =
            base64_decode(img["data"].get<std::string>());

        SlotImage img_sl;
        img_sl.id =
            img.count("id") != 0 ? img["id"].get<int>() : slot->images.size();
        img_sl.img_data = clip_image_u8_init();
        if (!clip_image_load_from_bytes(image_buffer.data(),
                                        image_buffer.size(), img_sl.img_data)) {
          LOG_DEBUG << "slot " << slot->id
                    << " - failed to load image [id: " << img_sl.id << "]";
          return false;
        }
        LOG_DEBUG << "slot " << slot->id << " - loaded image";
        img_sl.request_encode_image = true;
        slot->images.push_back(img_sl);
      }
      // process prompt
      // example: system prompt [img-102] user [img-103] describe [img-134] ->
      // [{id: 102, prefix: 'system prompt '}, {id: 103, prefix: ' user '},
      // {id: 134, prefix: ' describe '}]}
      if (slot->images.size() > 0 && !slot->prompt.is_array()) {
        std::string prompt = slot->prompt.get<std::string>();
        size_t pos = 0, begin_prefix = 0;
        std::string pattern = "[img-";
        while ((pos = prompt.find(pattern, pos)) != std::string::npos) {
          size_t end_prefix = pos;
          pos += pattern.length();
          size_t end_pos = prompt.find("]", pos);
          if (end_pos != std::string::npos) {
            std::string image_id = prompt.substr(pos, end_pos - pos);
            try {
              int img_id = std::stoi(image_id);
              bool found = false;
              for (SlotImage& img : slot->images) {
                if (img.id == img_id) {
                  found = true;
                  img.prefix_prompt =
                      prompt.substr(begin_prefix, end_prefix - begin_prefix);
                  begin_prefix = end_pos + 1;
                  break;
                }
              }
              if (!found) {
                LOG_WARN << "ERROR: Image with id: " << img_id
                         << ", not found.\n";
                slot->images.clear();
                return false;
              }
            } catch (const std::invalid_argument& e) {
              LOG_WARN << "Invalid image number id in prompt: " << e.what();
              slot->images.clear();
              return false;
            }
          }
        }
        slot->prompt = "";
        slot->params.input_suffix = prompt.substr(begin_prefix);
        slot->params.cache_prompt =
            false;  // multimodal doesn't support cache prompt
      }
    }
  }

  if (slot->smpl != nullptr) {
    gpt_sampler_free(slot->smpl);
  }
  slot->smpl = gpt_sampler_init(model, slot->sparams);
  // llama_set_rng_seed(ctx, slot->params.seed);
  slot->command = SlotCommand::kLoadPrompt;
  slot->prompt_tokens.clear();

  all_slots_are_idle = false;

  LOG_DEBUG << "slot " << slot->id
            << " is processing [task id: " << slot->task_id << "]";

  return true;
}

void LlamaServerContext::UpdateSystemPrompt() {
  system_tokens = ::llama_tokenize(ctx, system_prompt, add_bos_token);

  llama_batch_clear(batch);

  KvCacheClear();

  for (int i = 0; i < (int)system_tokens.size(); ++i) {
    llama_batch_add(batch, system_tokens[i], i, {0}, false);
  }

  if (llama_decode(ctx, batch) != 0) {
    LOG_WARN << __func__ << ": llama_decode() failed";
    return;
  }

  // assign the system KV cache to all parallel sequences
  for (int32_t i = 1; i < params.n_parallel; ++i) {
    llama_kv_cache_seq_cp(ctx, 0, i, 0, system_tokens.size());
  }

  LOG_DEBUG << "system prompt updated";
  system_need_update = false;
}

void LlamaServerContext::NotifySystemPromptChanged() {
  // release all slots
  for (LlamaClientSlot& slot : slots) {
    slot.Release();
  }

  system_need_update = true;
}

void LlamaServerContext::ProcessSystemPromptData(const json& sys_props) {
  system_prompt = sys_props.value("prompt", "");
  name_user = sys_props.value("anti_prompt", "");
  name_assistant = sys_props.value("assistant_name", "");

  if (slots.size() > 0) {
    NotifySystemPromptChanged();
  }
}

size_t LlamaServerContext::FindStoppingStrings(const std::string& text,
                                               const size_t last_token_size,
                                               const StopType type,
                                               LlamaClientSlot& slot) {
  size_t stop_pos = std::string::npos;

  for (const std::string& word : slot.params.antiprompt) {
    size_t pos;
    if (type == StopType::kStopFull) {
      const size_t tmp = word.size() + last_token_size;
      const size_t from_pos = text.size() > tmp ? text.size() - tmp : 0;
      pos = text.find(word, from_pos);
    } else {
      pos = find_partial_stop_string(word, text);
    }
    if (pos != std::string::npos &&
        (stop_pos == std::string::npos || pos < stop_pos)) {
      if (type == StopType::kStopFull) {
        slot.stopped_word = true;
        slot.stopping_word = word;
        slot.has_next_token = false;
      }
      stop_pos = pos;
    }
  }

  return stop_pos;
}

bool LlamaServerContext::ProcessToken(CompletionTokenOutput& result,
                                      LlamaClientSlot& slot) {
  // remember which tokens were sampled - used for repetition penalties during
  // sampling
  const std::string token_str = llama_token_to_piece(ctx, result.tok);
  slot.sampled = result.tok;

  // search stop word and delete it
  slot.generated_text += token_str;
  slot.has_next_token = true;

  // check if there is incomplete UTF-8 character at the end
  bool incomplete = false;
  for (unsigned i = 1; i < 5 && i <= slot.generated_text.size(); ++i) {
    unsigned char c = slot.generated_text[slot.generated_text.size() - i];
    if ((c & 0xC0) == 0x80) {
      // continuation byte: 10xxxxxx
      continue;
    }
    if ((c & 0xE0) == 0xC0) {
      // 2-byte character: 110xxxxx ...
      incomplete = i < 2;
    } else if ((c & 0xF0) == 0xE0) {
      // 3-byte character: 1110xxxx ...
      incomplete = i < 3;
    } else if ((c & 0xF8) == 0xF0) {
      // 4-byte character: 11110xxx ...
      incomplete = i < 4;
    }
    // else 1-byte character or invalid byte
    break;
  }

  if (!incomplete) {
    size_t pos = std::min(slot.sent_count, slot.generated_text.size());
    const std::string str_test = slot.generated_text.substr(pos);
    bool is_stop_full = false;
    size_t stop_pos = FindStoppingStrings(str_test, token_str.size(),
                                          StopType::kStopFull, slot);
    if (stop_pos != std::string::npos) {
      is_stop_full = true;
      slot.generated_text.erase(slot.generated_text.begin() + pos + stop_pos,
                                slot.generated_text.end());
      pos = std::min(slot.sent_count, slot.generated_text.size());
    } else {
      is_stop_full = false;
      stop_pos = FindStoppingStrings(str_test, token_str.size(),
                                     StopType::kStopPartial, slot);
    }

    // check if there is any token to predict
    if (stop_pos == std::string::npos ||
        (!slot.has_next_token && !is_stop_full && stop_pos > 0)) {
      // no send the stop word in the response
      result.text_to_send = slot.generated_text.substr(pos, std::string::npos);
      slot.sent_count += result.text_to_send.size();
      // add the token to slot queue and cache
    }
    slot.AddTokenString(result);
    if (slot.params.stream) {
      SendPartialResponse(slot, result);
    }
  }

  if (incomplete) {
    slot.has_next_token = true;
  }

  // check the limits
  if (slot.n_decoded > 2 && slot.has_next_token && !slot.HasBudget(params)) {
    slot.stopped_limit = true;
    slot.has_next_token = false;
  }

  if (llama_token_is_eog(model, result.tok)) {
    slot.stopped_eos = true;
    slot.has_next_token = false;
    LOG_VERBOSE("eos token found", {});
  }

  LOG_VERBOSE(
      "next token",
      {
          {"token", result.tok},
          {"token_text", tokens_to_output_formatted_string(ctx, result.tok)},
          {"has_next_token", slot.has_next_token},
          {"n_remain", slot.n_remaining},
          {"num_tokens_predicted", slot.n_decoded},
          {"stopped_eos", slot.stopped_eos},
          {"stopped_word", slot.stopped_word},
          {"stopped_limit", slot.stopped_limit},
          {"stopping_word", slot.stopping_word},
      });

  return slot.has_next_token;  // continue
}
bool LlamaServerContext::ProcessImages(LlamaClientSlot& slot) const {
  for (SlotImage& img : slot.images) {
    if (!img.request_encode_image) {
      continue;
    }

    if (!llava_image_embed_make_with_clip_img(
            clp_ctx, params.cpuparams.n_threads, img.img_data,
            &img.image_embedding, &img.image_tokens)) {
      LOG_WARN << "Error processing the given image";
      return false;
    }

    img.request_encode_image = false;
  }

  return slot.images.size() > 0;
}
void LlamaServerContext::SendError(TaskServer& task, std::string error) {
  SendError(task.id, task.multitask_id, error);
}

void LlamaServerContext::SendError(LlamaClientSlot& slot,
                                   const std::string& error) {
  SendError(slot.task_id, slot.multitask_id, error);
}

void LlamaServerContext::SendError(int id_task, int id_multi,
                                   const std::string& error) {
  TaskResult res;
  res.id = id_task;
  res.multitask_id = id_multi;
  res.stop = false;
  res.error = true;
  res.result_json = {{"content", error}};
  LOG_ERROR << "Internel error catched " << error;
  {
    std::lock_guard<std::mutex> lock(mutex_results);
    queue_results.push_back(res);
  }
  condition_results.notify_all();
}

void LlamaServerContext::AddMultiTask(int id, std::vector<int>& sub_ids) {
  TaskMulti multi;
  multi.id = id;
  std::copy(
      sub_ids.begin(), sub_ids.end(),
      std::inserter(multi.subtasks_remaining, multi.subtasks_remaining.end()));
  {
    std::lock_guard<std::mutex> lock(mutex_tasks);
    queue_multitasks.push_back(multi);
  }
  condition_tasks.notify_one();
}

void LlamaServerContext::UpdateMultiTask(int multitask_id, int subtask_id,
                                         TaskResult& result) {
  std::lock_guard<std::mutex> lock(mutex_tasks);
  for (auto& multitask : queue_multitasks) {
    if (multitask.id == multitask_id) {
      multitask.subtasks_remaining.erase(subtask_id);
      multitask.results.push_back(result);
      condition_tasks.notify_one();
    }
  }
}

json LlamaServerContext::GetFormatedGeneration(LlamaClientSlot& slot) {
  std::vector<std::string> samplers;
  samplers.reserve(slot.sparams.samplers.size());
  for (const auto& sampler : slot.sparams.samplers) {
    samplers.emplace_back(gpt_sampler_type_to_str(sampler));
  }

  return json{
      {"n_ctx", slot.n_ctx},
      {"model", params.model_alias},
      {"seed", slot.sparams.seed},
      {"temperature", slot.sparams.temp},
      {"dynatemp_range", slot.sparams.dynatemp_range},
      {"dynatemp_exponent", slot.sparams.dynatemp_exponent},
      {"top_k", slot.sparams.top_k},
      {"top_p", slot.sparams.top_p},
      {"min_p", slot.sparams.min_p},
      {"tfs_z", slot.sparams.tfs_z},
      {"typical_p", slot.sparams.typ_p},
      {"repeat_last_n", slot.sparams.penalty_last_n},
      {"repeat_penalty", slot.sparams.penalty_repeat},
      {"presence_penalty", slot.sparams.penalty_present},
      {"frequency_penalty", slot.sparams.penalty_freq},
      {"mirostat", slot.sparams.mirostat},
      {"mirostat_tau", slot.sparams.mirostat_tau},
      {"mirostat_eta", slot.sparams.mirostat_eta},
      {"penalize_nl", slot.sparams.penalize_nl},
      {"stop", slot.params.antiprompt},
      {"n_predict", slot.params.n_predict},
      {"n_keep", params.n_keep},
      {"ignore_eos", slot.sparams.ignore_eos},
      {"stream", slot.params.stream},
      //{"logit_bias",                slot.sparams.logit_bias},
      {"n_probs", slot.sparams.n_probs},
      {"min_keep", slot.sparams.min_keep},
      {"grammar", slot.sparams.grammar},
      {"samplers", samplers},
  };
}

void LlamaServerContext::SendPartialResponse(LlamaClientSlot& slot,
                                             CompletionTokenOutput tkn) {
  TaskResult res;
  res.id = slot.task_id;
  res.multitask_id = slot.multitask_id;
  res.error = false;
  res.stop = false;

  res.result_json = json{{"content", tkn.text_to_send},
                         {"stop", false},
                         {"slot_id", slot.id},
                         {"multimodal", multimodal}};

  if (slot.sparams.n_probs > 0) {
    std::vector<CompletionTokenOutput> probs_output = {};
    const std::vector<llama_token> to_send_toks =
        llama_tokenize(ctx, tkn.text_to_send, false);
    size_t probs_pos = std::min(slot.sent_token_probs_index,
                                slot.generated_token_probs.size());
    size_t probs_stop_pos =
        std::min(slot.sent_token_probs_index + to_send_toks.size(),
                 slot.generated_token_probs.size());
    if (probs_pos < probs_stop_pos) {
      probs_output = std::vector<CompletionTokenOutput>(
          slot.generated_token_probs.begin() + probs_pos,
          slot.generated_token_probs.begin() + probs_stop_pos);
    }
    slot.sent_token_probs_index = probs_stop_pos;
    res.result_json["completion_probabilities"] =
        probs_vector_to_json(ctx, probs_output);
  }

  if (slot.oaicompat) {
    res.result_json["oaicompat_token_ctr"] = slot.n_decoded;
    res.result_json["model"] = slot.oaicompat_model;
  }

  {
    std::lock_guard<std::mutex> lock(mutex_results);
    queue_results.push_back(res);
  }
  condition_results.notify_all();
}

void LlamaServerContext::SendFinalResponse(LlamaClientSlot& slot) {
  TaskResult res;
  res.id = slot.task_id;
  res.multitask_id = slot.multitask_id;
  res.error = false;
  res.stop = true;

  res.result_json =
      json{{"content", !slot.params.stream ? slot.generated_text : ""},
           {"slot_id", slot.id},
           {"stop", true},
           {"model", params.model_alias},
           {"tokens_predicted", slot.n_decoded},
           {"tokens_evaluated", slot.num_prompt_tokens},
           {"generation_settings", GetFormatedGeneration(slot)},
           {"prompt", slot.prompt},
           {"truncated", slot.truncated},
           {"stopped_eos", slot.stopped_eos},
           {"stopped_word", slot.stopped_word},
           {"stopped_limit", slot.stopped_limit},
           {"stopping_word", slot.stopping_word},
           {"tokens_cached", slot.n_past},
           {"timings", slot.GetFormatedTimings()}};

  if (slot.sparams.n_probs > 0) {
    std::vector<CompletionTokenOutput> probs = {};
    if (!slot.params.stream && slot.stopped_word) {
      const std::vector<llama_token> stop_word_toks =
          llama_tokenize(ctx, slot.stopping_word, false);
      probs = std::vector<CompletionTokenOutput>(
          slot.generated_token_probs.begin(),
          slot.generated_token_probs.end() - stop_word_toks.size());
    } else {
      probs = std::vector<CompletionTokenOutput>(
          slot.generated_token_probs.begin(),
          slot.generated_token_probs.begin() + slot.sent_token_probs_index);
    }
    res.result_json["completion_probabilities"] =
        probs_vector_to_json(ctx, probs);
  }

  if (slot.oaicompat) {
    res.result_json["oaicompat_token_ctr"] = slot.n_decoded;
    res.result_json["model"] = slot.oaicompat_model;
  }

  // parent multitask, if any, needs to be updated
  if (slot.multitask_id != -1) {
    UpdateMultiTask(slot.multitask_id, slot.task_id, res);
  }

  {
    std::lock_guard<std::mutex> lock(mutex_results);
    queue_results.push_back(res);
  }
  condition_results.notify_all();
}

void LlamaServerContext::SendEmbedding(LlamaClientSlot& slot) {
  TaskResult res;
  res.id = slot.task_id;
  res.multitask_id = slot.multitask_id;
  res.error = false;
  res.stop = true;

  const int n_embd = llama_n_embd(model);

  std::vector<float> embd_res(n_embd, 0.0f);

  for (int i = 0; i < batch.n_tokens; ++i) {
    if (!batch.logits[i] || batch.seq_id[i][0] != slot.id) {
      continue;
    }

    const float* embd = llama_get_embeddings_seq(ctx, batch.seq_id[i][0]);
    if (embd == NULL) {
      embd = llama_get_embeddings_ith(ctx, i);
    }

    if (embd == NULL) {
      LOG_ERROR << "failed to get embeddings" << " token " << batch.token[i]
                << ", seq_id " << batch.seq_id[i][0];

      res.result_json = json{
          {"embedding", std::vector<float>(n_embd, 0.0f)},
      };

      continue;
    }

    llama_embd_normalize(embd, embd_res.data(), n_embd);
  }
  res.result_json = json{
      {"tokens_evaluated", slot.num_prompt_tokens},
      {"embedding", embd_res},
  };

  {
    std::lock_guard<std::mutex> lock(mutex_results);
    queue_results.push_back(res);
  }
  condition_results.notify_all();
}

// for multiple images processing
bool LlamaServerContext::IngestImages(LlamaClientSlot& slot, int n_batch) {
  int image_idx = 0;

  while (image_idx < (int)slot.images.size()) {
    SlotImage& img = slot.images[image_idx];

    // process prefix prompt
    for (int32_t i = 0; i < (int32_t)batch.n_tokens; i += n_batch) {
      const int32_t n_tokens = std::min(n_batch, (int32_t)(batch.n_tokens - i));
      llama_batch batch_view = {
          n_tokens,
          batch.token + i,
          nullptr,
          batch.pos + i,
          batch.n_seq_id + i,
          batch.seq_id + i,
          batch.logits + i,
          0,
          0,
          0,  // unused
      };
      if (llama_decode(ctx, batch_view)) {
        LOG_WARN << __func__ << " : failed to eval\n";
        return false;
      }
    }

    // process image with llm
    for (int i = 0; i < img.image_tokens; i += n_batch) {
      int n_eval = img.image_tokens - i;
      if (n_eval > n_batch) {
        n_eval = n_batch;
      }

      const int n_embd = llama_n_embd(model);
      llama_batch batch_img = {
          n_eval,  nullptr,     (img.image_embedding + i * n_embd),
          nullptr, nullptr,     nullptr,
          nullptr, slot.n_past, 1,
          0,
      };
      if (llama_decode(ctx, batch_img)) {
        LOG_DEBUG << __func__ << " : failed to eval image";
        return false;
      }
      slot.n_past += n_eval;
    }
    image_idx++;

    llama_batch_clear(batch);

    // append prefix of next image
    const auto json_prompt =
        (image_idx >= (int)slot.images.size())
            ? slot.params.input_suffix
            :  // no more images, then process suffix prompt
            (json)(slot.images[image_idx].prefix_prompt);

    std::vector<llama_token> append_tokens =
        Tokenize(json_prompt, false);  // has next image
    for (int i = 0; i < (int)append_tokens.size(); ++i) {
      llama_batch_add(batch, append_tokens[i], slot.n_past, {slot.id}, true);
      slot.n_past += 1;
    }
  }

  return true;
}

int LlamaServerContext::SplitMultipromptTask(TaskServer& multiprompt_task) {
  int prompt_count = multiprompt_task.data.at("prompt").size();
  assert(prompt_count > 1);

  int multitask_id = id_gen++;
  std::vector<int> subtask_ids(prompt_count);
  for (int i = 0; i < prompt_count; i++) {
    json subtask_data = multiprompt_task.data;
    subtask_data["prompt"] = subtask_data["prompt"][i];

    // subtasks inherit everything else (infill mode, embedding mode, etc.)
    subtask_ids[i] =
        RequestCompletion(subtask_data, multiprompt_task.infill_mode,
                          multiprompt_task.embedding_mode, multitask_id);
  }

  // queue up the multitask so we can track its subtask progression
  AddMultiTask(multitask_id, subtask_ids);
  return multitask_id;
}

void LlamaServerContext::ProcessTasks() {
  while (true) {
    std::unique_lock<std::mutex> l(mutex_tasks);
    if (queue_tasks.empty()) {
      l.unlock();
      break;
    }

    TaskServer task = queue_tasks.front();

    if (task.type == TaskType::kCancelTask) {
      queue_tasks.erase(queue_tasks.begin());

      for (auto& slot : slots) {
        if (slot.task_id == task.target_id) {
          slot.Release();
          break;
        }
      }
      l.unlock();
    } else if (task.type == TaskType::kCompletionTask) {
      LlamaClientSlot* slot = GetSlot(json_value(task.data, "slot_id", -1));
      if (slot == nullptr) {
        l.unlock();
        return;
      }
      queue_tasks.erase(queue_tasks.begin());
      l.unlock();
      if (slot == nullptr) {
        LOG_WARN << "slot unavailable";
        // send error result
        SendError(task, "slot unavailable");
        return;
      }

      if (task.data.contains("system_prompt")) {
        ProcessSystemPromptData(task.data["system_prompt"]);
      }

      slot->Reset();

      slot->infill = task.infill_mode;
      slot->embedding = task.embedding_mode;
      slot->task_id = task.id;
      slot->multitask_id = task.multitask_id;

      if (!LaunchSlotWithData(slot, task.data)) {
        // send error result
        SendError(task, "internal_error");
        return;
      }
    }
  }

  // remove finished multitasks from the queue of multitasks, and add the
  // corresponding result to the result queue
  std::lock_guard<std::mutex> l(mutex_tasks);
  auto queue_iterator = queue_multitasks.begin();
  while (queue_iterator != queue_multitasks.end()) {
    if (queue_iterator->subtasks_remaining.empty()) {
      // all subtasks done == multitask is done
      TaskResult aggregate_result;
      aggregate_result.id = queue_iterator->id;
      aggregate_result.stop = true;
      aggregate_result.error = false;

      // collect json results into one json result
      std::vector<json> result_jsons;
      for (auto& subres : queue_iterator->results) {
        result_jsons.push_back(subres.result_json);
        aggregate_result.error = aggregate_result.error && subres.error;
      }
      aggregate_result.result_json = json{"results", result_jsons};

      {
        std::lock_guard<std::mutex> lock(mutex_results);
        queue_results.push_back(aggregate_result);
      }
      condition_results.notify_all();

      queue_iterator = queue_multitasks.erase(queue_iterator);
    } else {
      ++queue_iterator;
    }
  }
}

void LlamaServerContext::DoBackgroundTasks() {
  while (model_loaded_external) {
    UpdateSlots();
  }
  LOG_INFO << "Background task stopped! ";
  KvCacheClear();
  LOG_INFO << "KV cache cleared!";
}

bool LlamaServerContext::UpdateSlots() {
  // attend tasks
  ProcessTasks();

  // update the system prompt wait until all slots are idle state
  if (system_need_update && all_slots_are_idle) {
    LOG_DEBUG << "updating system prompt";
    UpdateSystemPrompt();
  }

  llama_batch_clear(batch);

  if (all_slots_are_idle) {
    if (system_prompt.empty() && clean_kv_cache) {
      LOG_DEBUG
          << "all slots are idle and system prompt is empty, clear the KV "
             "cache";
      KvCacheClear();
    }
    std::unique_lock<std::mutex> lock(mutex_tasks);
    condition_tasks.wait(lock, [&] {
      return (!queue_tasks.empty() && model_loaded_external) ||
             (queue_tasks.empty() && !model_loaded_external);
    });
  }

  for (LlamaClientSlot& slot : slots) {
    if (slot.IsProcessing() &&
        (int)system_tokens.size() + slot.n_past >= slot.n_ctx) {
      // Shift context
      const int n_left = slot.n_past - slot.params.n_keep - 1;
      const int n_discard = n_left / 2;

      LOG_DEBUG << "slot " << slot.id
                << " context shift - n_keep = " << slot.params.n_keep
                << ", n_left = " << n_left << ", n_discard: " << n_discard
                << ", n_ctx = " << n_ctx << ", n_past = " << slot.n_past
                << ", n_system_tokens = " << system_tokens.size()
                << ", n_cache_tokens = " << slot.cache_tokens.size();

      llama_kv_cache_seq_rm(ctx, slot.id, slot.params.n_keep + 1,
                            slot.params.n_keep + n_discard + 1);
      llama_kv_cache_seq_add(ctx, slot.id, slot.params.n_keep + 1 + n_discard,
                             slot.n_past, -n_discard);

      if (slot.params.cache_prompt) {
        for (size_t i = slot.params.n_keep + 1 + n_discard;
             i < slot.cache_tokens.size(); i++) {
          slot.cache_tokens[i - n_discard] = slot.cache_tokens[i];
        }

        slot.cache_tokens.resize(slot.cache_tokens.size() - n_discard);
      }

      slot.n_past -= n_discard;

      slot.truncated = true;
    }
  }

  // decode any currently ongoing sequences
  for (auto& slot : slots) {
    // release the slot
    if (slot.command == SlotCommand::kRelease) {
      slot.state = SlotState::kIdle;
      slot.command = SlotCommand::kNone;
      slot.t_last_used = ggml_time_us();

      LOG_INFO << "slot released: " << "id_slot: " << slot.id
               << ", id_task: " << slot.task_id << ", n_ctx: " << n_ctx
               << ", n_past: " << slot.n_past
               << ", n_system_tokens: " << system_tokens.size()
               << ", n_cache_tokens: " << slot.cache_tokens.size()
               << ", truncated: " << slot.truncated;

      continue;
    }

    if (slot.state == SlotState::kIdle) {
      continue;
    }

    slot.i_batch = batch.n_tokens;

    llama_batch_add(batch, slot.sampled, system_tokens.size() + slot.n_past,
                    {slot.id}, true);

    slot.n_decoded += 1;
    slot.n_past += 1;

    if (slot.params.cache_prompt) {
      slot.cache_tokens.push_back(slot.sampled);
    }

    LOG_TRACE << "slot decode token - " << " id_slot: " << slot.id
              << ", task_id: " << slot.task_id << ", n_ctx: " << n_ctx
              << ", n_past: " << slot.n_past
              << ", n_system_tokens: " << system_tokens.size()
              << ", n_cache_tokens: " << slot.cache_tokens.size()
              << ", truncated: " << slot.truncated;
  }

  // process in chunks of params.n_batch
  int32_t n_batch = llama_n_batch(ctx);
  int32_t n_ubatch = llama_n_ubatch(ctx);

  // assign workload to the slots
  if (params.cont_batching || batch.n_tokens == 0) {
    for (auto& slot : slots) {
      const bool has_prompt = slot.prompt.is_array() ||
                              (slot.prompt.is_string() &&
                               !slot.prompt.get<std::string>().empty()) ||
                              !slot.images.empty();

      // empty prompt passed -> release the slot and send empty response
      if (slot.state == SlotState::kIdle &&
          slot.command == SlotCommand::kLoadPrompt && !has_prompt) {
        slot.Release();
        slot.PrintTimings();
        SendFinalResponse(slot);
        continue;
      }

      // need process the prompt
      if (slot.state == SlotState::kIdle &&
          slot.command == SlotCommand::kLoadPrompt) {
        auto& prompt_tokens = slot.prompt_tokens;

        // we haven't tokenized the prompt yet - do it now:
        if (prompt_tokens.empty()) {
          slot.t_start_process_prompt = ggml_time_us();
          slot.t_start_genereration = 0;

          if (slot.infill) {
            bool suff_rm_leading_spc = true;
            if (params.input_suffix.find_first_of(' ') == 0 &&
                params.input_suffix.size() > 1) {
              params.input_suffix.erase(0, 1);
              suff_rm_leading_spc = false;
            }
            auto prefix_tokens = Tokenize(slot.params.input_prefix, false);
            auto suffix_tokens = Tokenize(slot.params.input_suffix, false);

            const int space_token =
                29871;  // TODO: this should not be hardcoded
            if (suff_rm_leading_spc && !suffix_tokens.empty() &&
                suffix_tokens[0] == space_token) {
              suffix_tokens.erase(suffix_tokens.begin());
            }

            prefix_tokens.insert(prefix_tokens.begin(),
                                 llama_token_prefix(model));
            prefix_tokens.insert(prefix_tokens.begin(),
                                 llama_token_bos(model));  // always add BOS
            prefix_tokens.insert(prefix_tokens.end(),
                                 llama_token_suffix(model));
            prefix_tokens.insert(prefix_tokens.end(), suffix_tokens.begin(),
                                 suffix_tokens.end());
            prefix_tokens.push_back(llama_token_middle(model));
            prompt_tokens = prefix_tokens;
          } else {
            prompt_tokens = Tokenize(
                slot.prompt,
                system_prompt.empty() &&
                    add_bos_token);  // add BOS if there isn't system prompt
          }

          slot.n_past = 0;
          slot.num_prompt_tokens = prompt_tokens.size();

          LOG_VERBOSE(
              "prompt tokenized",
              {
                  {"id_slot", slot.id},
                  {"id_task", slot.task_id},
                  {"n_ctx", slot.n_ctx},
                  {"n_keep", slot.params.n_keep},
                  {"n_prompt_tokens", slot.num_prompt_tokens},
                  {"prompt_tokens", tokens_to_str(ctx, prompt_tokens.cbegin(),
                                                  prompt_tokens.cend())},
              });

          if (slot.embedding) {
            // this prompt is too large to process - discard it
            if (slot.num_prompt_tokens > n_ubatch) {
              LOG_DEBUG << "This prompt is too large to process: "
                           "num_promt_tokens = "
                        << slot.num_prompt_tokens
                        << ", n_ubatch = " << n_ubatch;
              slot.state = SlotState::kProcessing;
              slot.command = SlotCommand::kNone;
              slot.Release();
              slot.PrintTimings();
              SendFinalResponse(slot);
              continue;
            }
          } else {
            if (slot.params.n_keep < 0) {
              slot.params.n_keep = slot.num_prompt_tokens;
            }
            slot.params.n_keep = std::min(slot.n_ctx - 4, slot.params.n_keep);

            // if input prompt is too big, truncate it
            if (slot.num_prompt_tokens >= slot.n_ctx) {
              const int n_left = slot.n_ctx - slot.params.n_keep;
              const int n_block_size = n_left / 2;
              const int erased_blocks =
                  (slot.num_prompt_tokens - slot.params.n_keep - n_block_size) /
                  n_block_size;

              std::vector<llama_token> new_tokens(
                  prompt_tokens.begin(),
                  prompt_tokens.begin() + slot.params.n_keep);
              new_tokens.insert(new_tokens.end(),
                                prompt_tokens.begin() + slot.params.n_keep +
                                    erased_blocks * n_block_size,
                                prompt_tokens.end());

              LOG_VERBOSE("input truncated",
                          {
                              {"id_slot", slot.id},
                              {"id_task", slot.task_id},
                              {"n_ctx", slot.n_ctx},
                              {"n_keep", slot.params.n_keep},
                              {"n_left", n_left},
                              {"n_prompt_tokens", slot.num_prompt_tokens},
                              {"prompt_tokens",
                               tokens_to_str(ctx, prompt_tokens.cbegin(),
                                             prompt_tokens.cend())},
                          });

              slot.truncated = true;
              prompt_tokens = new_tokens;

              slot.num_prompt_tokens = prompt_tokens.size();
              GGML_ASSERT(slot.num_prompt_tokens < slot.n_ctx);
            }

            gpt_sampler_reset(slot.smpl);

            if (!slot.params.cache_prompt) {
              slot.n_past = 0;
              slot.num_prompt_tokens_processed = slot.num_prompt_tokens;
            } else {
              // push the prompt into the sampling context (do not apply grammar)
              for (auto& token : prompt_tokens) {
                gpt_sampler_accept(slot.smpl, token, false);
              }

              slot.n_past = common_part(slot.cache_tokens, prompt_tokens);
              slot.num_prompt_tokens_processed =
                  slot.num_prompt_tokens - slot.n_past;

              LOG_TEE("slot %d : in cache: %i tokens | to process: %i tokens\n",
                      slot.id, slot.n_past, slot.num_prompt_tokens_processed);
            }
          }

          if (slot.n_past == slot.num_prompt_tokens) {
            // we have to evaluate at least 1 token to generate logits.
            LOG_DEBUG << "slot " << slot.id
                      << " : we have to evaluate at least 1 token to "
                         "generate logits";
            slot.n_past--;
          }

          slot.num_prompt_tokens_processed = 0;
        }

        if (slot.embedding) {
          // cannot fit the prompt in the current batch - will try next iter
          if (batch.n_tokens + slot.num_prompt_tokens > n_batch) {
            continue;
          }
        }

        LOG_VERBOSE(
            "prompt ingested",
            {
                {"n_past", slot.n_past},
                {"cached",
                 tokens_to_str(ctx, slot.cache_tokens.cbegin(),
                               slot.cache_tokens.cbegin() + slot.n_past)},
                {"to_eval",
                 tokens_to_str(ctx, slot.cache_tokens.cbegin() + slot.n_past,
                               slot.cache_tokens.cend())},
            });

        // keep only the common part
        int p0 = (int)system_tokens.size() + slot.n_past;
        if (!llama_kv_cache_seq_rm(ctx, slot.id, p0, -1)) {
          // could not partially delete (likely using a non-Transformer model)
          llama_kv_cache_seq_rm(ctx, slot.id, -1, -1);

          p0 = (int)system_tokens.size();
          if (p0 != 0) {
            // copy over the system prompt when there is one
            llama_kv_cache_seq_cp(ctx, 0, slot.id, -1, -1);
          }

          // there is no common part left (except for the system prompt)
          slot.n_past = 0;
          // TODO: is the system prompt ever in the sampling context?
          gpt_sampler_reset(slot.smpl);
        }

        // remove the non-common part from the cache
        slot.cache_tokens.resize(slot.n_past);
        LOG_INFO << "kv cache rm [p0, end) - " << " id_slot: " << slot.id
                 << ", task_id: " << slot.task_id << ", p0: " << p0;

        const bool has_images = ProcessImages(slot);

        // process the prefix of first image
        std::vector<llama_token> prefix_tokens =
            has_images ? Tokenize(slot.images[0].prefix_prompt, add_bos_token)
                       : prompt_tokens;
        for (; slot.n_past < (int)prefix_tokens.size(); ++slot.n_past) {
          llama_batch_add(batch, prefix_tokens[slot.n_past],
                          system_tokens.size() + slot.n_past, {slot.id}, false);
          if (slot.params.cache_prompt) {
            slot.cache_tokens.push_back(prompt_tokens[slot.n_past]);
          }
          slot.num_prompt_tokens_processed++;
        }

        LOG_VERBOSE("prompt processing progress",
                    {
                        {"id_slot", slot.id},
                        {"n_past", slot.n_past},
                        {"n_ctx", n_ctx},
                        {"n_tokens", batch.n_tokens},
                        {"progress", (float)slot.num_prompt_tokens_processed /
                                         slot.num_prompt_tokens},
                    });

        if (has_images && !IngestImages(slot, n_batch)) {
          LOG_WARN << "failed processing images";
          slot.state = SlotState::kProcessing;
          slot.command = SlotCommand::kNone;
          slot.Release();
          SendError(slot, "Failed processing images");
          return false;
        }

        // entire prompt has been processed - start decoding new tokens
        if (has_images || slot.n_past == slot.num_prompt_tokens) {
          slot.state = SlotState::kProcessing;
          slot.command = SlotCommand::kNone;

          GGML_ASSERT(batch.n_tokens > 0);

          // extract the logits only for the last token
          if (batch.n_tokens > 0) {
            batch.logits[batch.n_tokens - 1] = true;
          }

          slot.n_decoded = 0;
          slot.i_batch = batch.n_tokens - 1;

          LOG_VERBOSE("prompt done", {
                                         {"id_slot", slot.id},
                                         {"n_past", slot.n_past},
                                         {"n_ctx", n_ctx},
                                         {"n_tokens", batch.n_tokens},
                                     });
        }
      }
    }
  }

  if (batch.n_tokens == 0) {
    all_slots_are_idle = true;
    return true;
  }

  for (int32_t i = 0; i < (int32_t)batch.n_tokens; i += n_batch) {
    const int32_t n_tokens = std::min(n_batch, (int32_t)(batch.n_tokens - i));
    llama_batch batch_view = {
        n_tokens,
        batch.token + i,
        nullptr,
        batch.pos + i,
        batch.n_seq_id + i,
        batch.seq_id + i,
        batch.logits + i,
        0,
        0,
        0,  // unused
    };

    const int ret = llama_decode(ctx, batch_view);
    if (ret != 0) {
      if (n_batch == 1 || ret < 0) {
        // if you get here, it means the KV cache is full - try increasing it via the context size
        LOG_ERROR << "Failed to decode the batch: KV cache is full - try "
                     "increasing it via the context size: "
                  << "i = " << i << ", n_batch = " << n_batch
                  << ", ret = " << ret;
        for (auto& slot : slots) {
          slot.state = SlotState::kProcessing;
          slot.command = SlotCommand::kNone;
          slot.Release();
          SendError(slot,
                    "Input prompt is too big compared to KV size. Please "
                    "try increasing KV size.");
        }
        break;  // break loop of n_batch
      }

      LOG_WARN << "Failed to find free space in the KV cache, retrying with "
                  "smaller n_batch = "
               << n_batch / 2;

      // retry with half the batch size to try to find a free slot in the KV
      // cache
      n_batch /= 2;
      i -= n_batch;
      continue;
    }

    for (auto& slot : slots) {
      if (slot.i_batch < (int)i || slot.i_batch >= (int)(i + n_tokens)) {
        continue;
      }

      // prompt evaluated for embedding
      if (slot.embedding) {
        SendEmbedding(slot);
        slot.Release();
        slot.i_batch = -1;
        return true;
      }

      CompletionTokenOutput result;
      const llama_token id =
          gpt_sampler_sample(slot.smpl, ctx, slot.i_batch - i);

      gpt_sampler_accept(slot.smpl, id, true);

      if (slot.n_decoded == 1) {
        slot.t_start_genereration = ggml_time_us();
        slot.t_prompt_processing =
            (slot.t_start_genereration - slot.t_start_process_prompt) / 1e3;
      }

      const auto* cur_p = gpt_sampler_get_candidates(slot.smpl);
      result.tok = id;

      for (size_t i = 0; i < (size_t)slot.sparams.n_probs; ++i) {
        result.probs.push_back({
            cur_p->data[i].id,
            i >= cur_p->size ? 0.0f : cur_p->data[i].p,
        });
      }

      if (!ProcessToken(result, slot)) {
        slot.Release();
        slot.PrintTimings();
        SendFinalResponse(slot);
      }

      slot.i_batch = -1;
    }
  }
  return true;
}
