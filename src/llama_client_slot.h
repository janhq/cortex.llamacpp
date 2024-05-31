#pragma once

#include <string>
#include <vector>

#include "common.h"
#include "json.hpp"
#include "llama.h"
#include "llava.h"
#include "stb_image.h"
#include "trantor/utils/Logger.h"

#include "clip.h"

static bool server_verbose = false;

#ifndef SERVER_VERBOSE
#define SERVER_VERBOSE 1
#endif

#if SERVER_VERBOSE != 1
#define LOG_VERBOSE(MSG, ...)
#else
#define LOG_VERBOSE(MSG, ...)                                      \
  do {                                                             \
    if (server_verbose) {                                          \
      server_log("VERBOSE", __func__, __LINE__, MSG, __VA_ARGS__); \
    }                                                              \
  } while (0)
#endif

#define LOG_ERROR_LLAMA(MSG, ...) \
  server_log("ERROR", __func__, __LINE__, MSG, __VA_ARGS__)
#define LOG_WARNING_LLAMA(MSG, ...) \
  server_log("WARNING", __func__, __LINE__, MSG, __VA_ARGS__)
#define LOG_INFO_LLAMA(MSG, ...) \
  server_log("INFO", __func__, __LINE__, MSG, __VA_ARGS__)

using json = nlohmann::json;

enum class SlotState: uint8_t {
  kIdle,
  kProcessing,
};

enum class SlotCommand: uint8_t {
  kNone,
  kLoadPrompt,
  kRelease,
};

struct SlotParams {
  bool stream = true;
  bool cache_prompt =
      false;  // remember the prompt to avoid reprocessing all prompt

  uint32_t seed = -1;      // RNG seed
  int32_t n_keep = 0;      // number of tokens to keep from initial prompt
  int32_t n_predict = -1;  // new tokens to predict

  std::vector<std::string> antiprompt;

  json input_prefix;
  json input_suffix;
};

struct SlotImage {
  int32_t id;

  bool request_encode_image = false;
  float* image_embedding = nullptr;
  int32_t image_tokens = 0;

  clip_image_u8* img_data;

  std::string prefix_prompt;  // before of this image
};

struct CompletionTokenOutput {
  struct TokenProb {
    llama_token tok;
    float prob;
  };

  std::vector<TokenProb> probs;
  llama_token tok;
  std::string text_to_send;
};

struct LlamaClientSlot {
  int id;
  int task_id = -1;

  struct SlotParams params;

  SlotState state = SlotState::kIdle;
  SlotCommand command = SlotCommand::kNone;

  // used to determine the slot that has been used the longest
  int64_t t_last_used = -1;

  // generation props
  int32_t n_ctx = 0;  // context size per slot
  int32_t n_past = 0;
  int32_t n_decoded = 0;
  int32_t n_remaining = -1;
  int32_t i_batch = -1;
  int32_t n_predict   = -1; // TODO: disambiguate from params.n_predict

  int32_t num_prompt_tokens = 0;
  int32_t num_prompt_tokens_processed = 0;

  json prompt;

  // when a task is submitted, we first tokenize the prompt and store it here
  std::vector<llama_token> prompt_tokens;

  std::string generated_text;
  llama_token sampled;
  std::vector<llama_token> cache_tokens;
  std::vector<CompletionTokenOutput> generated_token_probs;

  bool infill = false;
  bool embedding = false;
  bool has_next_token = true;
  bool truncated = false;
  bool stopped_eos = false;
  bool stopped_word = false;
  bool stopped_limit = false;

  bool oaicompat = false;
  std::string oaicompat_model;

  std::string stopping_word;

  // sampling
  struct llama_sampling_params sparams;
  llama_sampling_context* ctx_sampling = nullptr;

  // multimodal
  std::vector<SlotImage> images;

  // stats
  size_t sent_count = 0;
  size_t sent_token_probs_index = 0;

  int64_t t_start_process_prompt;
  int64_t t_start_genereration;

  double t_prompt_processing;  // ms
  double t_token_generation;   // ms

  // multitasks
  int multitask_id = -1;

  void Reset();

  bool HasBudget(gpt_params& global_params);

  bool Available() const;

  bool IsProcessing() const;

  void AddTokenString(const CompletionTokenOutput& token);

  void Release();

  json GetFormatedTimings();

  void PrintTimings() const;
};
