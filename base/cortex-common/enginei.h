#pragma once

#include <filesystem>
#include <functional>
#include <memory>
#include <vector>

#include "json/value.h"
#include "trantor/utils/Logger.h"

// Interface for inference engine.
// Note: only append new function to keep the compatibility.
class EngineI {
 public:
  struct EngineLoadOption {
    // engine
    std::filesystem::path engine_path;
    std::filesystem::path deps_path;
    bool is_custom_engine_path;

    // logging
    std::filesystem::path log_path;
    int max_log_lines;
    trantor::Logger::LogLevel log_level;
  };

  struct EngineUnloadOption {
    // place holder for now
  };

  virtual ~EngineI() {}

  virtual void Load(EngineLoadOption opts) = 0;

  virtual void Unload(EngineUnloadOption opts) = 0;

  virtual void HandleChatCompletion(
      std::shared_ptr<Json::Value> json_body,
      std::function<void(Json::Value&&, Json::Value&&)>&& callback) = 0;
  virtual void HandleEmbedding(
      std::shared_ptr<Json::Value> json_body,
      std::function<void(Json::Value&&, Json::Value&&)>&& callback) = 0;
  virtual void LoadModel(
      std::shared_ptr<Json::Value> json_body,
      std::function<void(Json::Value&&, Json::Value&&)>&& callback) = 0;
  virtual void UnloadModel(
      std::shared_ptr<Json::Value> json_body,
      std::function<void(Json::Value&&, Json::Value&&)>&& callback) = 0;
  virtual void GetModelStatus(
      std::shared_ptr<Json::Value> json_body,
      std::function<void(Json::Value&&, Json::Value&&)>&& callback) = 0;

  // For backward compatible checking, add to list when we add more APIs
  virtual bool IsSupported(const std::string& f) {
    if (f == "HandleChatCompletion" || f == "HandleEmbedding" ||
        f == "LoadModel" || f == "UnloadModel" || f == "GetModelStatus" ||
        f == "GetModels" || f == "SetFileLogger" || f == "SetLogLevel" ||
        f == "StopInferencing") {
      return true;
    }
    return false;
  }

  // API to get running models.
  virtual void GetModels(
      std::shared_ptr<Json::Value> json_body,
      std::function<void(Json::Value&&, Json::Value&&)>&& callback) = 0;
  // API for set file logger
  virtual void SetFileLogger(int max_log_lines,
                             const std::string& log_path) = 0;
  virtual void SetLogLevel(trantor::Logger::LogLevel log_level) = 0;

  virtual void StopInferencing(const std::string& model_id) = 0;
};
