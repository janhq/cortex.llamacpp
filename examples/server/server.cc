#include "cortex-common/enginei.h"
#include "dylib.h"
#include "httplib.h"
#include "json/reader.h"

#include <signal.h>
#include <condition_variable>
#include <mutex>
#include <queue>
#include "trantor/utils/Logger.h"

class Server {
 public:
  Server() {
    dylib_ = std::make_unique<dylib>("./engines/cortex.llamacpp", "engine");
    auto func = dylib_->get_function<EngineI*()>("get_engine");
    engine_ = func();
  }

  ~Server() {
    if (engine_) {
      delete engine_;
    }
  }

 public:
  std::unique_ptr<dylib> dylib_;
  EngineI* engine_;

  struct SyncQueue {
    void push(std::pair<Json::Value, Json::Value>&& p) {
      std::unique_lock<std::mutex> l(mtx);
      q.push(p);
      cond.notify_one();
    }

    std::pair<Json::Value, Json::Value> wait_and_pop() {
      std::unique_lock<std::mutex> l(mtx);
      cond.wait(l, [this] { return !q.empty(); });
      auto res = q.front();
      q.pop();
      return res;
    }

    std::mutex mtx;
    std::condition_variable cond;
    // Status and result
    std::queue<std::pair<Json::Value, Json::Value>> q;
  };
};

std::function<void(int)> shutdown_handler;
std::atomic_flag is_terminating = ATOMIC_FLAG_INIT;

inline void signal_handler(int signal) {
  if (is_terminating.test_and_set()) {
    // in case it hangs, we can force terminate the server by hitting Ctrl+C twice
    // this is for better developer experience, we can remove when the server is stable enough
    fprintf(stderr, "Received second interrupt, terminating immediately.\n");
    exit(1);
  }

  shutdown_handler(signal);
}

using SyncQueue = Server::SyncQueue;

int main(int argc, char** argv) {
  std::string hostname = "127.0.0.1";
  int port = 3928;
  if (argc > 1) {
    hostname = argv[1];
  }

  // Check for port argument
  if (argc > 2) {
    port = std::atoi(argv[2]);  // Convert string argument to int
  }

  Server server;
  Json::Reader r;
  auto svr = std::make_unique<httplib::Server>();

  if (!svr->bind_to_port(hostname, port)) {
    fprintf(stderr, "\ncouldn't bind to server socket: hostname=%s port=%d\n\n",
            hostname.c_str(), port);
    return 1;
  }

  auto process_non_stream_res = [&server](httplib::Response& resp,
                                          SyncQueue& q) {
    auto [status, res] = q.wait_and_pop();
    resp.set_content(res.toStyledString().c_str(),
                     "application/json; charset=utf-8");
    resp.status = status["status_code"].asInt();
  };

  auto process_stream_res = [&server](httplib::Response& resp,
                                      std::shared_ptr<SyncQueue> q) {
    const auto chunked_content_provider =
        [&server, q](size_t size, httplib::DataSink& sink) {
          while (true) {
            auto [status, res] = q->wait_and_pop();
            auto str = res["data"].asString();
            LOG_TRACE << "data: " << str;

            if (!sink.write(str.c_str(), str.size())) {
              LOG_WARN << "Failed to write";
              //   return false;
            }
            if (status["has_error"].asBool() || status["is_done"].asBool()) {
              LOG_INFO << "Done";
              sink.done();
              break;
            }
          }

          return true;
        };
    resp.set_chunked_content_provider("text/event-stream",
                                      chunked_content_provider,
                                      [](bool) { LOG_INFO << "Done"; });
  };

  const auto handle_load_model = [&](const httplib::Request& req,
                                     httplib::Response& resp) {
    resp.set_header("Access-Control-Allow-Origin",
                    req.get_header_value("Origin"));
    auto req_body = std::make_shared<Json::Value>();
    r.parse(req.body, *req_body);
    server.engine_->LoadModel(
        req_body, [&server, &resp](Json::Value status, Json::Value res) {
          resp.set_content(res.toStyledString().c_str(),
                           "application/json; charset=utf-8");
          resp.status = status["status_code"].asInt();
        });
  };

  const auto handle_unload_model = [&](const httplib::Request& req,
                                       httplib::Response& resp) {
    resp.set_header("Access-Control-Allow-Origin",
                    req.get_header_value("Origin"));
    auto req_body = std::make_shared<Json::Value>();
    r.parse(req.body, *req_body);
    server.engine_->UnloadModel(
        req_body, [&server, &resp](Json::Value status, Json::Value res) {
          resp.set_content(res.toStyledString().c_str(),
                           "application/json; charset=utf-8");
          resp.status = status["status_code"].asInt();
        });
  };

  const auto handle_completions = [&](const httplib::Request& req,
                                      httplib::Response& resp) {
    resp.set_header("Access-Control-Allow-Origin",
                    req.get_header_value("Origin"));
    auto req_body = std::make_shared<Json::Value>();
    r.parse(req.body, *req_body);
    bool is_stream = (*req_body).get("stream", false).asBool();
    // This is an async call, need to use queue
    auto q = std::make_shared<SyncQueue>();
    server.engine_->HandleChatCompletion(
        req_body, [&server, q](Json::Value status, Json::Value res) {
          q->push(std::make_pair(status, res));
        });
    if (is_stream) {
      process_stream_res(resp, q);
    } else {
      process_non_stream_res(resp, *q);
    }
  };

  const auto handle_embeddings = [&](const httplib::Request& req,
                                     httplib::Response& resp) {
    resp.set_header("Access-Control-Allow-Origin",
                    req.get_header_value("Origin"));
    auto req_body = std::make_shared<Json::Value>();
    r.parse(req.body, *req_body);
    // This is an async call, need to use queue
    SyncQueue q;
    server.engine_->HandleEmbedding(
        req_body, [&server, &q](Json::Value status, Json::Value res) {
          q.push(std::make_pair(status, res));
        });
    process_non_stream_res(resp, q);
  };

  const auto handle_get_model_status = [&](const httplib::Request& req,
                                           httplib::Response& resp) {
    resp.set_header("Access-Control-Allow-Origin",
                    req.get_header_value("Origin"));
    auto req_body = std::make_shared<Json::Value>();
    r.parse(req.body, *req_body);
    server.engine_->GetModelStatus(
        req_body, [&server, &resp](Json::Value status, Json::Value res) {
          resp.set_content(res.toStyledString().c_str(),
                           "application/json; charset=utf-8");
          resp.status = status["status_code"].asInt();
        });
  };

  const auto handle_get_running_models = [&](const httplib::Request& req,
                                             httplib::Response& resp) {
    resp.set_header("Access-Control-Allow-Origin",
                    req.get_header_value("Origin"));
    auto req_body = std::make_shared<Json::Value>();
    r.parse(req.body, *req_body);
    server.engine_->GetModels(
        req_body, [&server, &resp](Json::Value status, Json::Value res) {
          resp.set_content(res.toStyledString().c_str(),
                           "application/json; charset=utf-8");
          resp.status = status["status_code"].asInt();
        });
  };

  svr->Post("/loadmodel", handle_load_model);
  // Use POST since httplib does not read request body for GET method
  svr->Post("/unloadmodel", handle_unload_model);
  svr->Post("/v1/chat/completions", handle_completions);
  svr->Post("/v1/embeddings", handle_embeddings);
  svr->Post("/modelstatus", handle_get_model_status);
  svr->Get("/models", handle_get_running_models);
  std::atomic<bool> running = true;
  svr->Delete("/destroy",
            [&](const httplib::Request& req, httplib::Response& resp) {
              LOG_INFO << "Received Stop command";
              running = false;
            });

  LOG_INFO << "HTTP server listening: " << hostname << ":" << port;
  svr->new_task_queue = [] {
    return new httplib::ThreadPool(5);
  };
  // run the HTTP server in a thread - see comment below
  std::thread t([&]() {
    if (!svr->listen_after_bind()) {
      return 1;
    }

    return 0;
  });

  shutdown_handler = [&](int) {
    running = false;
  };
#if defined(__unix__) || (defined(__APPLE__) && defined(__MACH__))
  struct sigaction sigint_action;
  sigint_action.sa_handler = signal_handler;
  sigemptyset(&sigint_action.sa_mask);
  sigint_action.sa_flags = 0;
  sigaction(SIGINT, &sigint_action, NULL);
  sigaction(SIGTERM, &sigint_action, NULL);
#elif defined(_WIN32)
  auto console_ctrl_handler = +[](DWORD ctrl_type) -> BOOL {
    return (ctrl_type == CTRL_C_EVENT) ? (signal_handler(SIGINT), true) : false;
  };
  SetConsoleCtrlHandler(
      reinterpret_cast<PHANDLER_ROUTINE>(console_ctrl_handler), true);
#endif

  while (running) {
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
  }

  svr->stop();
  t.join();
  LOG_DEBUG << "Server shutdown";
  return 0;
}