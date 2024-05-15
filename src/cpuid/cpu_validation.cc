#include "cpu_validation.h"
#include "cpu_info.h"
#include "vulkan_util_init.h"
namespace cpuid::llamacpp {

bool IsSupportedVulka() {
#if defined(CORTEX_VULKAN)
  struct sample_info info = {};
  init_global_layer_properties(info);

  /* VULKAN_KEY_START */

  // initialize the VkApplicationInfo structure
  VkApplicationInfo app_info = {};
  app_info.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
  app_info.pNext = NULL;
  app_info.pApplicationName = APP_SHORT_NAME;
  app_info.applicationVersion = 1;
  app_info.pEngineName = APP_SHORT_NAME;
  app_info.engineVersion = 1;
  app_info.apiVersion = VK_API_VERSION_1_0;

  // initialize the VkInstanceCreateInfo structure
  VkInstanceCreateInfo inst_info = {};
  inst_info.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
  inst_info.pNext = NULL;
  inst_info.flags = 0;
  inst_info.pApplicationInfo = &app_info;
  inst_info.enabledExtensionCount = 0;
  inst_info.ppEnabledExtensionNames = NULL;
  inst_info.enabledLayerCount = 0;
  inst_info.ppEnabledLayerNames = NULL;

  VkInstance inst;
  VkResult res;

  res = vkCreateInstance(&inst_info, NULL, &inst);
  if (res == VK_ERROR_INCOMPATIBLE_DRIVER) {
    std::cout << "cannot find a compatible Vulkan ICD\n";
    return false;
  } else if (res) {
    std::cout << "unknown error\n";
    return false;
  }

  vkDestroyInstance(inst, NULL);
  return true;
#endif
  return false;
}

// TODO implement Result for better perf
std::pair<bool, std::string> IsValidInstructions() {
  cpuid::CpuInfo info;
#if defined(_WIN32)
#if defined(CORTEX_AVX512)
  auto res = info.has_avx512_f() || info.has_avx512_dq() ||
             info.has_avx512_ifma() || info.has_avx512_pf() ||
             info.has_avx512_er() || info.has_avx512_cd() ||
             info.has_avx512_bw() || info.has_avx512_vl() ||
             info.has_avx512_vbmi() || info.has_avx512_vbmi2() ||
             info.has_avx512_vnni() || info.has_avx512_bitalg() ||
             info.has_avx512_vpopcntdq() || info.has_avx512_4vnniw() ||
             info.has_avx512_4fmaps() || info.has_avx512_vp2intersect();
  return res ? std::make_pair(true, "")
             : std::make_pair(false, "System does not support AVX512");
#elif defined(CORTEX_AVX2)
  return info.has_avx2()
             ? std::make_pair(true, "")
             : std::make_pair(false, "System does not support AVX2");
#elif defined(CORTEX_VULKAN)
  return IsSupportedVulka()
             ? std::make_pair(true, "")
             : std::make_pair(false, "System does not support VULKA");
#else
  return info.has_avx() ? std::make_pair(true, "")
                        : std::make_pair(false, "System does not support AVX");
#endif
#elif defined(__APPLE__)
  return std::make_pair(true, "");
#else
#if defined(CORTEX_CUDA)
  return std::make_pair(true, "");
#elif defined(CORTEX_AVX512)
  auto res = info.has_avx512_f() || info.has_avx512_dq() ||
             info.has_avx512_ifma() || info.has_avx512_pf() ||
             info.has_avx512_er() || info.has_avx512_cd() ||
             info.has_avx512_bw() || info.has_avx512_vl() ||
             info.has_avx512_vbmi() || info.has_avx512_vbmi2() ||
             info.has_avx512_vnni() || info.has_avx512_bitalg() ||
             info.has_avx512_vpopcntdq() || info.has_avx512_4vnniw() ||
             info.has_avx512_4fmaps() || info.has_avx512_vp2intersect();
  return res ? std::make_pair(true, "")
             : std::make_pair(false, "System does not support AVX512");
#elif defined(CORTEX_AVX2)
  return info.has_avx2()
             ? std::make_pair(true, "")
             : std::make_pair(false, "System does not support AVX2");
#elif defined(CORTEX_VULKAN)
  return IsSupportedVulka()
             ? std::make_pair(true, "")
             : std::make_pair(false, "System does not support VULKA");
#else
  return info.has_avx() ? std::make_pair(true, "")
                        : std::make_pair(false, "System does not support AVX");
#endif
#endif
  return std::make_pair(true, "");
}
}  // namespace cpuid::llamacpp