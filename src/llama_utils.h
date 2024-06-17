#pragma once
#include <algorithm>
#include <cstdio>
#include <fstream>
#include <iostream>
#include <ostream>
#include <random>
#include <regex>
#include <string>
#include <vector>

namespace llama_utils {

inline std::string models_folder = "./models";

inline std::string extractBase64(const std::string& input) {
  std::regex pattern("base64,(.*)");
  std::smatch match;

  if (std::regex_search(input, match, pattern)) {
    std::string base64_data = match[1];
    base64_data = base64_data.substr(0, base64_data.length() - 1);
    return base64_data;
  }

  return "";
}

// Helper function to encode data to Base64
inline std::string base64Encode(const std::vector<unsigned char>& data) {
  static const char encodingTable[] =
      "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
  std::string encodedData;
  int i = 0;
  int j = 0;
  unsigned char array3[3];
  unsigned char array4[4];

  for (unsigned char c : data) {
    array3[i++] = c;
    if (i == 3) {
      array4[0] = (array3[0] & 0xfc) >> 2;
      array4[1] = ((array3[0] & 0x03) << 4) + ((array3[1] & 0xf0) >> 4);
      array4[2] = ((array3[1] & 0x0f) << 2) + ((array3[2] & 0xc0) >> 6);
      array4[3] = array3[2] & 0x3f;

      for (i = 0; i < 4; i++)
        encodedData += encodingTable[array4[i]];
      i = 0;
    }
  }

  if (i) {
    for (j = i; j < 3; j++)
      array3[j] = '\0';

    array4[0] = (array3[0] & 0xfc) >> 2;
    array4[1] = ((array3[0] & 0x03) << 4) + ((array3[1] & 0xf0) >> 4);
    array4[2] = ((array3[1] & 0x0f) << 2) + ((array3[2] & 0xc0) >> 6);

    for (j = 0; j < i + 1; j++)
      encodedData += encodingTable[array4[j]];

    while (i++ < 3)
      encodedData += '=';
  }

  return encodedData;
}

// Function to load an image and convert it to Base64
inline std::string imageToBase64(const std::string& imagePath) {
  std::ifstream imageFile(imagePath, std::ios::binary);
  if (!imageFile.is_open()) {
    throw std::runtime_error("Could not open the image file.");
  }

  std::vector<unsigned char> buffer(std::istreambuf_iterator<char>(imageFile),
                                    {});
  return base64Encode(buffer);
}

// Helper function to generate a unique filename
inline std::string generateUniqueFilename(const std::string& prefix,
                                          const std::string& extension) {
  // Get current time as a timestamp
  auto now = std::chrono::system_clock::now();
  auto now_ms = std::chrono::time_point_cast<std::chrono::milliseconds>(now);
  auto epoch = now_ms.time_since_epoch();
  auto value = std::chrono::duration_cast<std::chrono::milliseconds>(epoch);

  // Generate a random number
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_int_distribution<> dis(1000, 9999);

  std::stringstream ss;
  ss << prefix << value.count() << "_" << dis(gen) << extension;
  return ss.str();
}

inline void processLocalImage(
    const std::string& localPath,
    std::function<void(const std::string&)> callback) {
  try {
    std::string base64Image = imageToBase64(localPath);
    callback(base64Image);  // Invoke the callback with the Base64 string
  } catch (const std::exception& e) {
    std::cerr << "Error during processing: " << e.what() << std::endl;
  }
}

inline std::string rtrim(const std::string& str) {
  size_t end = str.find_last_not_of("\n\t ");
  return (end == std::string::npos) ? "" : str.substr(0, end + 1);
}

inline std::string generate_random_string(std::size_t length) {
  const std::string characters =
      "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz";

  std::random_device rd;
  std::mt19937 generator(rd());

  std::uniform_int_distribution<> distribution(
      0, static_cast<int>(characters.size()) - 1);

  std::string random_string(length, '\0');
  std::generate_n(random_string.begin(), length,
                  [&]() { return characters[distribution(generator)]; });

  return random_string;
}

inline void ltrim(std::string& s) {
  s.erase(s.begin(), std::find_if(s.begin(), s.end(), [](unsigned char ch) {
            return !std::isspace(ch);
          }));
};

inline std::string GetModelId(const Json::Value& jsonBody) {
  // First check if model exists in request
  if (!jsonBody["model"].isNull()) {
    return jsonBody["model"].asString();
  } else if (!jsonBody["model_alias"].isNull()) {
    return jsonBody["model_alias"].asString();
  }

  // We check llama_model_path and model_path for loadmodel request
  auto input_v0 = jsonBody["llama_model_path"];
  auto input_v1 = jsonBody["model_path"];
  auto input = input_v0.isNull() ? input_v1 : input_v0;
  if (!input.isNull()) {
    auto s = input.asString();
    std::replace(s.begin(), s.end(), '\\', '/');
    auto const pos = s.find_last_of('/');
    // We only truncate the extension if file name has gguf extension
    if (s.substr(s.find_last_of('.') + 1) == "gguf") {
      return s.substr(pos + 1, s.find_last_of('.') - pos - 1);
    } else {
      return s.substr(pos + 1);
    }
  }
  return {};
}

}  // namespace llama_utils