#pragma once

#include "task_result.h"
#include <functional>

class EngineI {
 public:
  virtual ~EngineI() {}

  virtual void HandleChatCompletion(
      std::shared_ptr<Json::Value> jsonBody,
      std::function<void(TaskResult&&)>&& callback) = 0;
  virtual void HandleEmbedding(
      std::shared_ptr<Json::Value> jsonBody,
      std::function<void(TaskResult&&)>&& callback) = 0;
  virtual void LoadModel(
      std::shared_ptr<Json::Value> jsonBody,
      std::function<void(TaskResult&&)>&& callback) = 0;
  virtual void UnloadModel(
      std::shared_ptr<Json::Value> jsonBody,
      std::function<void(TaskResult&&)>&& callback) = 0;
  virtual void GetModelStatus(
      std::shared_ptr<Json::Value> jsonBody,
      std::function<void(TaskResult&&)>&& callback) = 0;
};