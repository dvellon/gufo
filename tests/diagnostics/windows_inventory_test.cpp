#include <cassert>
#include <iostream>
#include <string>

#include "src/core/diagnostics/fingerprint.h"
#include "src/core/diagnostics/linux_sysfs.h"
#include "src/core/diagnostics/system_inventory.h"

int main() {
  const gufo::diagnostics::LinuxSysfs host;
  const auto cpu = host.QueryCpuInfo();
  const auto memory = host.QueryMemInfo();
  const auto os = host.QueryUname();

  assert(cpu && !cpu->model_name.empty() && cpu->logical_cores > 0);
  assert(memory && memory->total_bytes > 0 && memory->available_bytes > 0);
  assert(os && os->sysname == "Windows" && !os->release.empty());

  const auto inventory = gufo::diagnostics::CollectSystemInventory(host);
  assert(inventory.cpu.source == "CPUID + Win32");
  assert(inventory.memory.source == "GlobalMemoryStatusEx");
  assert(inventory.toolchain.amdgpu_status ==
         "not applicable (Windows HIP runtime)");

  const std::string inventory_text = inventory.ToHuman();
  assert(inventory_text.find("Windows Version") != std::string::npos);
  assert(inventory_text.find("Linux Kernel") == std::string::npos);

  const auto fingerprint =
      gufo::diagnostics::GenerateMachineFingerprint(inventory);
  const std::string fingerprint_text = fingerprint.ToHuman();
  assert(fingerprint_text.find("x86_64-windows") != std::string::npos);
  assert(fingerprint_text.find("Loaded Runtime") != std::string::npos);
  assert(fingerprint_text.find("x86_64-linux") == std::string::npos);

  std::cout << "Windows CPU, memory, OS and fingerprint inventory passed\n";
}
