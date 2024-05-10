#pragma once

#include "cpu_info_impl.h"

namespace cpuid {
inline namespace STEINWURF_CPUID_VERSION {
void init_cpuinfo(CpuInfo::Impl& info) {
  // Visual Studio 2012 (and above) guarantees the NEON capability when
  // compiling for Windows Phone 8 (and above)

#if defined(PLATFORM_WINDOWS_PHONE)
  info.has_neon = true;
#else
  info.has_neon = false;
#endif
}
}  // namespace STEINWURF_CPUID_VERSION
}  // namespace cpuid