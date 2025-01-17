#pragma once
#include <json.hpp>
#include "json/value.h"
#include "sampling.h"
namespace llama::inferences {

Json::Value ConvertNlohmannToJsonCpp(const nlohmann::json& input) {
  if (input.is_null()) {
    return Json::Value(Json::nullValue);
  } else if (input.is_boolean()) {
    return Json::Value(input.get<bool>());
  } else if (input.is_number_integer()) {
    return Json::Value(input.get<int>());
  } else if (input.is_number_float()) {
    return Json::Value(input.get<double>());
  } else if (input.is_string()) {
    return Json::Value(input.get<std::string>());
  } else if (input.is_array()) {
    Json::Value arr(Json::arrayValue);
    for (const auto& elem : input) {
      arr.append(ConvertNlohmannToJsonCpp(elem));
    }
    return arr;
  } else if (input.is_object()) {
    Json::Value obj(Json::objectValue);
    for (auto it = input.begin(); it != input.end(); ++it) {
      obj[it.key()] = ConvertNlohmannToJsonCpp(it.value());
    }
    return obj;
  }
  return Json::Value(Json::nullValue);
}

nlohmann::json ConvertJsonCppToNlohmann(const Json::Value& input) {
  if (input.isNull()) {
    return nullptr;
  } else if (input.isBool()) {
    return input.asBool();
  } else if (input.isInt()) {
    return input.asInt();
  } else if (input.isDouble()) {
    return input.asDouble();
  } else if (input.isString()) {
    return input.asString();
  } else if (input.isArray()) {
    nlohmann::json arr = nlohmann::json::array();
    for (const auto& elem : input) {
      arr.push_back(ConvertJsonCppToNlohmann(elem));
    }
    return arr;
  } else if (input.isObject()) {
    nlohmann::json obj = nlohmann::json::object();
    for (const auto& key : input.getMemberNames()) {
      obj[key] = ConvertJsonCppToNlohmann(input[key]);
    }
    return obj;
  }
  return nullptr;
}
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
  std::string prompt;

  int seed = -1;
  float dynatemp_range = 0.0f;
  float dynatemp_exponent = 1.0f;
  int top_k = 40;
  float min_p = 0.05f;
  float typ_p = 1.0f;
  int repeat_last_n = 64;
  float penalty_repeat = 1.0f;
  bool mirostat = false;
  float mirostat_tau = 5.0f;
  float mirostat_eta = 0.1f;
  bool penalize_nl = false;
  bool ignore_eos = false;
  bool logprobs = false;
  int n_probs = 0;
  int min_keep = 0;
  int n = 1;
  bool include_usage = false;
  std::string grammar;
  Json::Value logit_bias = Json::Value(Json::arrayValue);
  Json::Value json_schema;

  static Json::Value ConvertLogitBiasToArray(const Json::Value& input) {
    Json::Value result(Json::arrayValue);
    if (input.isObject()) {
      const auto& memberNames = input.getMemberNames();
      for (const auto& tokenStr : memberNames) {
        Json::Value pair(Json::arrayValue);
        pair.append(std::stoi(tokenStr));
        pair.append(input[tokenStr].asFloat());
        result.append(pair);
      }
    }
    return result;
  }
};

inline ChatCompletionRequest fromJson(std::shared_ptr<Json::Value> jsonBody) {
  ChatCompletionRequest completion;
  common_params_sampling default_params;
  if (jsonBody) {
    completion.stream = (*jsonBody).get("stream", false).asBool();
    if (completion.stream) {
      auto& stream_options = (*jsonBody)["stream_options"];
      if (!stream_options.isNull()) {
        completion.include_usage =
            stream_options.get("include_usage", false).asBool();
      }
    }
    completion.max_tokens = (*jsonBody).get("max_tokens", 500).asInt();
    completion.top_p = (*jsonBody).get("top_p", 0.95).asFloat();
    completion.temperature = (*jsonBody).get("temperature", 0.8).asFloat();
    completion.frequency_penalty =
        (*jsonBody).get("frequency_penalty", 0).asFloat();
    completion.presence_penalty =
        (*jsonBody).get("presence_penalty", 0).asFloat();
    completion.messages = (*jsonBody)["messages"];
    completion.prompt = jsonBody->get("prompt", "").asString();
    completion.stop = (*jsonBody)["stop"];
    completion.model_id = (*jsonBody).get("model", {}).asString();

    completion.seed = (*jsonBody).get("seed", -1).asInt();
    completion.dynatemp_range =
        (*jsonBody).get("dynatemp_range", 0.0f).asFloat();
    completion.dynatemp_exponent =
        (*jsonBody).get("dynatemp_exponent", 0.0f).asFloat();
    completion.top_k = (*jsonBody).get("top_k", 40).asInt();
    completion.min_p = (*jsonBody).get("min_p", 0.05f).asFloat();
    completion.typ_p = (*jsonBody).get("typ_p", 1.0f).asFloat();
    completion.repeat_last_n = (*jsonBody).get("repeat_last_n", 64).asInt();
    completion.penalty_repeat =
        (*jsonBody).get("repeat_penalty", 1.1f).asFloat();
    completion.mirostat = (*jsonBody).get("mirostat", false).asBool();
    completion.mirostat_tau = (*jsonBody).get("mirostat_tau", 5.0f).asFloat();
    completion.mirostat_eta = (*jsonBody).get("mirostat_eta", 0.1f).asFloat();
    completion.penalize_nl = (*jsonBody).get("penalize_nl", true).asBool();
    completion.ignore_eos = (*jsonBody).get("ignore_eos", false).asBool();
    completion.logprobs = (*jsonBody).get("logprobs", false).asBool();
    int top_logprobs = (*jsonBody).get("top_logprobs", 0).asInt();
    int n_probs = (*jsonBody).get("n_probs", 0).asInt();
    if (completion.logprobs) {
      completion.n_probs = top_logprobs;
    } else if (n_probs > 0) {
      completion.logprobs = true;
      completion.n_probs = n_probs;
    }
    completion.min_keep = (*jsonBody).get("min_keep", 0).asInt();
    completion.n = (*jsonBody).get("n", 1).asInt();
    completion.grammar = (*jsonBody).get("grammar", "").asString();
    completion.json_schema = (*jsonBody).get("response_format", Json::Value::null);
    const Json::Value& input_logit_bias = (*jsonBody)["logit_bias"];
    if (!input_logit_bias.isNull()) {
      completion.logit_bias =
          ChatCompletionRequest::ConvertLogitBiasToArray(input_logit_bias);
    }
  }
  return completion;
}
}  // namespace llama::inferences