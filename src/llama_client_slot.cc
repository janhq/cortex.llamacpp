#include "llama_client_slot.h"

void LlamaClientSlot::Reset() {
  num_prompt_tokens = 0;
  generated_text = "";
  truncated = false;
  stopped_eos = false;
  stopped_word = false;
  stopped_limit = false;
  stopping_word = "";
  n_past = 0;
  sent_count = 0;
  sent_token_probs_index = 0;
  infill = false;

  generated_token_probs.clear();

  for (SlotImage& img : images) {
    free(img.image_embedding);
    if (img.img_data) {
      clip_image_u8_free(img.img_data);
    }
    img.prefix_prompt = "";
  }

  images.clear();
}

bool LlamaClientSlot::HasBudget(common_params& global_params) {
  n_remaining = -1;
  if (params.n_predict != -1) {
    n_remaining = params.n_predict - n_decoded;
  } else if (global_params.n_predict != -1) {
    n_remaining = global_params.n_predict - n_decoded;
  }
  return n_remaining > 0 || n_remaining == -1;  // no budget || limitless
}

bool LlamaClientSlot::Available() const {
  return state == SlotState::kIdle && command == SlotCommand::kNone;
}

bool LlamaClientSlot::IsProcessing() const {
  return (state == SlotState::kIdle && command == SlotCommand::kLoadPrompt) ||
         state == SlotState::kProcessing;
}

void LlamaClientSlot::AddTokenString(const CompletionTokenOutput& token) {
  if (command == SlotCommand::kRelease) {
    return;
  }
  generated_token_probs.push_back(token);
}

void LlamaClientSlot::Release() {
  if (state == SlotState::kIdle || state == SlotState::kProcessing) {
    t_token_generation = (ggml_time_us() - t_start_genereration) / 1e3;
    command = SlotCommand::kRelease;
  }
}

json LlamaClientSlot::GetFormatedTimings() {
  return json{
      {"prompt_n", num_prompt_tokens_processed},
      {"prompt_ms", t_prompt_processing},
      {"prompt_per_token_ms",
       t_prompt_processing / num_prompt_tokens_processed},
      {"prompt_per_second",
       1e3 / t_prompt_processing * num_prompt_tokens_processed},

      {"predicted_n", n_decoded},
      {"predicted_ms", t_token_generation},
      {"predicted_per_token_ms", t_token_generation / n_decoded},
      {"predicted_per_second", 1e3 / t_token_generation * n_decoded},
  };
}

void LlamaClientSlot::PrintTimings() const {
  LOG_DEBUG << __func__ << ": prompt eval time = " << t_prompt_processing
            << "ms / " << num_prompt_tokens_processed << " tokens ("
            << t_prompt_processing / num_prompt_tokens_processed
            << " ms per "
               "token, "
            << 1e3 / t_prompt_processing * num_prompt_tokens_processed
            << " tokens per second)";
  LOG_DEBUG << __func__ << ":        eval time = " << t_token_generation
            << " ms / " << n_decoded << " runs   ("
            << t_token_generation / n_decoded
            << " ms per "
               "token, "
            << 1e3 / t_token_generation * n_decoded << " tokens per second)\n";
  LOG_DEBUG << __func__ << ":       total time = "
            << t_prompt_processing + t_token_generation << " ms";
}
