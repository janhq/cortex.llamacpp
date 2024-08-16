#include <iostream>
#include "CLI11.hpp"
#define CPPHTTPLIB_OPENSSL_SUPPORT
#include <filesystem>
#include <list>
#include <string>
#include "httplib.h"
#include "json.hpp"
#include "yaml-cpp/yaml.h"
namespace fs = std::filesystem;

using namespace nlohmann::literals;

int main(int argc, char** argv) {
  if (argc == 1) {
    STARTUPINFOW si;
    PROCESS_INFORMATION pi;
    ZeroMemory(&si, sizeof(si));
    si.cb = sizeof(si);
    ZeroMemory(&pi, sizeof(pi));

    if (!CreateProcessW(
            L"E:\\workspace\\cortex.llamacpp\\examples\\server\\test\\server."
            L"exe",  // the path to the executable file
            L"",     // command line arguments passed to the child
            NULL, NULL, FALSE, 0, NULL, NULL, &si, &pi)) {
      std::cout << "Could not start server" << std::endl;
    } else {
      std::cout << "Start server" << std::endl;
    }
    return 0;
  }

  //   YAML::Node config;
  //   try {
  //     config = YAML::LoadFile(
  //         "E:/workspace/cortex.llamacpp/examples/server/test/tinyllama/"
  //         "model.yml");
  //   } catch (const std::exception& e) {
  //     std::cerr << e.what() << '\n';
  //   }

  //   if (config["stop"]) {
  //     auto s = config["stop"].as<std::list<std::string>>();
  //     // for (auto i : s) {
  //     //   std::cout << i << std::endl;
  //     // }
  //   } else {
  //     std::cout << "Missed stop field" << std::endl;
  //   }

    // if (config["prompt_template"]) {
    //   auto pt = config["prompt_template"].as<std::string>();
    //   auto system_pr = pt.substr(0, pt.find_first_of('{'));
    //   // std::cout << system_pr << std::endl;
    //   auto user_pr = pt.substr(pt.find_first_of('}') + 1,
    //                            pt.find_last_of('{') - pt.find_first_of('}') - 1);
    //   // std::cout << user_pr << std::endl;
    //   auto ast_pr = pt.substr(pt.find_last_of('}') + 1);
    //   // std::cout << ast_pr << std::endl;
    // }

  CLI::App app{"App description"};
  argv = app.ensure_utf8(argv);

  // cortex models list
  std::string path =
      "E:/workspace/cortex.llamacpp/examples/server/build/Release";
  std::vector<std::string> supported_models;
  for (const auto& entry : fs::directory_iterator(path)) {
    if (entry.path().string().find(".yaml") != std::string::npos) {
      // std::cout << entry.path() << std::endl;
      auto node = YAML::LoadFile(entry.path().string());
      // std::cout << node["name"].as<std::string>() << std::endl;
      supported_models.push_back(node["name"].as<std::string>());
    }
  }

  auto models_cmd =
      app.add_subcommand("models", "Subcommands for managing models");
  //   std::string cmd = "";
  //   models_cmd->add_option("list", cmd, "List all available local models");
  auto list_cmd =
      models_cmd->add_subcommand("list", "List all available local models");
  list_cmd->callback([&]() {
    for (auto const& m : supported_models) {
      std::cout << m << std::endl;
    }
  });
  auto pull_cmd = models_cmd->add_subcommand("get", "Get a model by ID");

  std::string model_id;
  pull_cmd->add_option("model_id", model_id, "A help string");

  httplib::Client hf_cli("https://huggingface.co");
  pull_cmd->callback([&]() {
    if (model_id.empty()) {
      std::cout << "Please input model" << std::endl;
    } else {
      std::cout << "Start to pull model: " << model_id << std::endl;

    //   std::string body;
    //   std::ofstream m_yml;
    //   m_yml.open("model.yml");

    //   auto res = hf_cli.Get("/cortexso/tinyllama/resolve/main/model.yml",
    //                         [&](const char* data, size_t data_length) {
    //                           // std::cout << "test" << std::endl;
    //                           body.append(data, data_length);
    //                           m_yml.write(data, data_length);
    //                           return true;
    //                         });
    //   m_yml.close();
    //   std::cout << body << std::endl;

      std::ofstream gguf_file;
      gguf_file.open("model.gguf", std::ios::binary);
      
      //   https://huggingface.co/cortexso/tinyllama/resolve/main/model.gguf
      //https://huggingface.co/cortexso/tinyllama/resolve/main/model.gguf
      hf_cli.set_follow_location(true);
      uint64_t last = 0;
      uint64_t tot = 0;
      hf_cli.Get(
          "/cortexso/tinyllama/resolve/main/model.gguf",
          [](const httplib::Response &res ){
            if(res.status != httplib::StatusCode::OK_200) {
                std::cout << "HTTP error: " << res.reason << std::endl;
                return false;
            }
            return true;
          },
          [&](const char* data, size_t data_length) {
            tot += data_length;
            gguf_file.write(data, data_length);
            return true;
          },
          [&last](uint64_t current, uint64_t total) {
            if(current - last > 100000000) {
                last = current;
                std::cout << "Downloading: " << current << " / " << total << std::endl;
            }
            if(current == total) {
                std::cout << "Done download: " << static_cast<double>(total)/1024/1024 << " MiB" << std::endl;
                return false;
            }
            return true;
          });
      gguf_file.close();
      std::cout << "Done: " << tot << std::endl;
    }
  });

  auto run_cmd = app.add_subcommand("run", "Shortcut to start a model");
  std::string model_name = "";
  run_cmd->add_option("model", model_name, "Run a model help string");
  run_cmd->callback([&]() {
    if (model_name.empty()) {
      std::cout << "model empty" << std::endl;
    } else {
      httplib::Client cli("127.0.0.1:3928");
      nlohmann::json json_data;
      json_data["model_path"] =
          "E:/workspace/cortex.llamacpp/examples/server/"
          "build/Release/tinyllama/model.gguf";
      json_data["model"] = "tinyllama";
      json_data["system_prompt"] = "<|system|>\n";
      json_data["user_prompt"] = "<|user|>\n";
      json_data["ai_prompt"] = "<|assistant|>";
      auto data_str = json_data.dump();

      auto res = cli.Post("/loadmodel", httplib::Headers(), data_str.data(),
                          data_str.size(), "application/json");
      if (res) {
        if (res->status == httplib::StatusCode::OK_200) {
          std::cout << res->body << std::endl;
        }
      } else {
        auto err = res.error();
        std::cout << "HTTP error: " << httplib::to_string(err) << std::endl;
      }
    }
  });

  auto chat_cmd = app.add_subcommand("chat", "Send a chat request to a model");
  chat_cmd->callback([&]() {
    bool running = true;

    while (true) {
      std::string user_input;
      std::cout << "> ";
      std::getline(std::cin, user_input);
      if (user_input == "exit()") {
        httplib::Client cli("127.0.0.1:3928");
        auto res = cli.Delete("/destroy");
        if (res) {

        } else {
          auto err = res.error();
          std::cout << "HTTP error: " << httplib::to_string(err) << std::endl;
        }
        break;
      }

      if (!user_input.empty()) {
        struct ChunkParser {
          std::string content;
          bool is_done = false;

          ChunkParser(const char* data, size_t data_length) {
            if (data && data_length > 6) {
              std::string s(data + 6, data_length - 6);
              if (s.find("[DONE]") != std::string::npos) {
                is_done = true;
              } else {
                nlohmann::json ex = nlohmann::json::parse(s);
                // std::cout << "dump " << ex["choices"][0]["delta"]["content"].dump().c_str() << std::endl;
                content = ex["choices"][0]["delta"]["content"].dump();
                // json dump append "" to data
                if (content.size() >= 2) {
                  content = content.substr(1, content.size() - 2);
                  // std::cout << content  << " " << content.size() << std::endl;
                  // TODO special character parsing
                  if (content == "\\n") {
                    // std::cout << "Find new line" << std::endl;
                    content = '\n';
                  }
                }
              }
            }
          }
        };
        httplib::Client cli("127.0.0.1:3928");
        nlohmann::json json_data;
        nlohmann::json msg;
        msg["role"] = "user";
        msg["content"] = user_input;
        json_data["messages"] = nlohmann::json::array({msg});
        json_data["model"] = "tinyllama";
        json_data["stream"] = true;
        json_data["stop"] = {"</s>"};
        auto data_str = json_data.dump();
        // std::cout << data_str << std::endl;
        cli.set_read_timeout(std::chrono::seconds(60));
        // std::cout << "> ";
        httplib::Request req;
        req.method = "POST";
        req.path = "/v1/chat/completions";
        req.body = data_str;
        req.content_receiver = [&](const char* data, size_t data_length,
                                   uint64_t offset, uint64_t total_length) {
          // std::string s(data, data_length);
          // std::cout << s << std::endl;
          ChunkParser cp(data, data_length);
          if (cp.is_done) {
            std::cout << std::endl;
            return false;
          }
          std::cout << cp.content;
          return true;
        };
        cli.send(req);
      }
      //  std::cout << "ok Done" << std::endl;
    }
    // std::cout << "Done" << std::endl;
  });
  CLI11_PARSE(app, argc, argv);

  return 0;

  //   auto pull_cmd =
  //       app.add_subcommand("pull",
  //                          "Download a model from a registry. Working with "
  //                          "HuggingFace repositories. For available models, "
  //                          "please visit https://huggingface.co/cortexso");

  //   std::string filename = "";
  //   pull_cmd->add_option("model_id", filename, "A help string");

  //   pull_cmd->callback([&]() {
  //     if (filename.empty()) {
  //       std::cout << "Please input model" << std::endl;
  //     } else {
  //       std::cout << "Start to pull model: " << filename << std::endl;
  //       //   auto host = "www.httpwatch.com";
  //       //   auto port = 80;
  //       httplib::Client cli("https://huggingface.co");
  //       // std::cout << "1111" << std::endl;
  //       std::string body;
  //       // std::cout << "adfsf" << std::endl;
  //       auto res = cli.Get("/cortexso/tinyllama/resolve/main/model.yml",
  //                          [&](const char* data, size_t data_length) {
  //                            // std::cout << "test" << std::endl;
  //                            body.append(data, data_length);
  //                            return true;
  //                          });
  //       std::cout << body << std::endl;
  //     }
  //   });

  //   CLI11_PARSE(app, argc, argv);
  //   return 0;
}