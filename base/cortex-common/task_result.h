#pragma once
#include "json/value.h"

enum class HttpCode: int {
    k200OK = 200,
    k400BadRequest = 400,
    k409Conflict = 409,
    k500InternalServerError = 500
};
struct TaskResult {
    bool is_done;
    bool has_error;
    bool is_stream;
    HttpCode status = HttpCode::k200OK;
    Json::Value result_json;
};