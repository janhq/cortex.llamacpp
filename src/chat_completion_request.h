#pragma once
#include "json/value.h"

namespace llama::inferences {
struct ChatCompletionRequest {
  bool stream = false;
  int max_tokens = 500;
  float top_p = 0.95f;
  float temperature = 0.8f;
  float frequency_penalty = 0;
  float presence_penalty = 0;
  Json::Value stop = Json::Value(Json::arrayValue);
  Json::Value messages = Json::Value(Json::arrayValue);
  std::string model_id;
};

inline ChatCompletionRequest fromJson(std::shared_ptr<Json::Value> jsonBody) {
  ChatCompletionRequest completion;
  if (jsonBody) {
    completion.stream = (*jsonBody).get("stream", false).asBool();
    completion.max_tokens = (*jsonBody).get("max_tokens", 500).asInt();
    completion.top_p = (*jsonBody).get("top_p", 0.95).asFloat();
    completion.temperature = (*jsonBody).get("temperature", 0.8).asFloat();
    completion.frequency_penalty =
        (*jsonBody).get("frequency_penalty", 0).asFloat();
    completion.presence_penalty =
        (*jsonBody).get("presence_penalty", 0).asFloat();
    completion.messages = (*jsonBody)["messages"];
    completion.stop = (*jsonBody)["stop"];
    completion.model_id = (*jsonBody).get("model", {}).asString();
  }
  return completion;
}
}  // namespace llama::inferences