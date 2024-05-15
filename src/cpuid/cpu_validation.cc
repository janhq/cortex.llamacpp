#include "cpu_validation.h"
#include "cpu_info.h"

namespace cpuid::llamacpp {

bool IsValidInstructions() {
  cpuid::CpuInfo info;
#if defined(_WIN32)
#if defined(LLAMA_AVX512)
  return info.has_avx512_f() || info.has_avx512_dq() ||
         info.has_avx512_ifma() || info.has_avx512_pf() ||
         info.has_avx512_er() || info.has_avx512_cd() || info.has_avx512_bw() ||
         info.has_avx512_vl() || info.has_avx512_vbmi() ||
         info.has_avx512_vbmi2() || info.has_avx512_vnni() ||
         info.has_avx512_bitalg() || info.has_avx512_vpopcntdq() ||
         info.has_avx512_4vnniw() || info.has_avx512_4fmaps() ||
         info.has_avx512_vp2intersect();
#elif defined(LLAMA_AVX2)
  return info.has_avx2();
#elif defined(LLAMA_VULKAN)
  return true;
#else
  return info.has_avx();
#endif
#elif defined(__APPLE__)
  return true;
#else
#if defined(LLAMA_CUDA)
  return true;
#elif defined(LLAMA_AVX512)
  return info.has_avx512_f() || info.has_avx512_dq() ||
         info.has_avx512_ifma() || info.has_avx512_pf() ||
         info.has_avx512_er() || info.has_avx512_cd() || info.has_avx512_bw() ||
         info.has_avx512_vl() || info.has_avx512_vbmi() ||
         info.has_avx512_vbmi2() || info.has_avx512_vnni() ||
         info.has_avx512_bitalg() || info.has_avx512_vpopcntdq() ||
         info.has_avx512_4vnniw() || info.has_avx512_4fmaps() ||
         info.has_avx512_vp2intersect();
#elif defined(LLAMA_AVX2)
  return info.has_avx2();
#elif defined(LLAMA_VULKAN)
  return true;
#else
  return info.has_avx();
#endif
#endif
  return true;
}
}  // namespace cpuid::llamacpp