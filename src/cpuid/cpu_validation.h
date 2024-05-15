#pragma once
#include <string>
#include <utility>

namespace cpuid::llamacpp {
std::pair<bool, std::string> IsValidInstructions();
}  // namespace cpuid