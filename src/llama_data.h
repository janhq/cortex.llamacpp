#pragma once
#include "json/json.h"

struct IsDone {
  bool is_done;
  int operator()() { return is_done; }
};

struct HasError {
  bool has_error;
  int operator()() { return has_error; }
};

struct IsStream {
  bool is_stream;
  int operator()() { return is_stream; }
};

struct StatusCode {
  int status_code;
  int operator()() { return status_code; }
};

struct ResStatus {
 private:
  IsDone is_done;
  HasError has_error;
  IsStream is_stream;
  StatusCode status_code;

 public:
  ResStatus(IsDone is_done, HasError has_error, IsStream is_stream,
            StatusCode status_code)
      : is_done(is_done),
        has_error(has_error),
        is_stream(is_stream),
        status_code(status_code) {}

  Json::Value ToJson() {
    Json::Value status;
    status["is_done"] = is_done();
    status["has_error"] = has_error();
    status["is_stream"] = is_stream();
    status["status_code"] = status_code();
    return status;
  };
};

struct ResStreamData {
 private:
  std::string s;

 public:
  ResStreamData(std::string s) : s(std::move(s)) {}
  Json::Value ToJson() {
    Json::Value d;
    d["data"] = s;
    return d;
  }
};