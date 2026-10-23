# Resolve the compile-time HIP capability contract. Gufo's production runtime
# remains specialized for gfx1151; gfx942 is exposed only to narrowly scoped
# experimental targets that state and check their requirements.
function(gufo_resolve_hip_architecture architecture allow_gfx942)
  if (architecture MATCHES ";")
    message(FATAL_ERROR
      "Gufo requires exactly one HIP architecture per build, got: ${architecture}")
  endif()

  if (architecture STREQUAL "gfx1151")
    set(_family "strix-halo")
    set(_wave_size 32)
    set(_has_wmma_wave32 1)
    set(_has_mfma_wave64 0)
    set(_production_supported 1)
    set(_experimental 0)
  elseif (architecture STREQUAL "gfx942" AND allow_gfx942)
    set(_family "cdna3")
    set(_wave_size 64)
    set(_has_wmma_wave32 0)
    set(_has_mfma_wave64 1)
    set(_production_supported 0)
    set(_experimental 1)
  elseif (architecture STREQUAL "gfx942")
    message(FATAL_ERROR
      "gfx942 is available only to the explicit experimental FlashMap backend")
  else()
    message(FATAL_ERROR
      "Gufo production kernels require gfx1151; unsupported architecture: ${architecture}")
  endif()

  set(GUFO_HIP_ARCHITECTURE "${architecture}" PARENT_SCOPE)
  set(GUFO_HIP_ARCHITECTURE_FAMILY "${_family}" PARENT_SCOPE)
  set(GUFO_HIP_WAVE_SIZE "${_wave_size}" PARENT_SCOPE)
  set(GUFO_HIP_HAS_WMMA_WAVE32 "${_has_wmma_wave32}" PARENT_SCOPE)
  set(GUFO_HIP_HAS_MFMA_WAVE64 "${_has_mfma_wave64}" PARENT_SCOPE)
  set(GUFO_HIP_PRODUCTION_SUPPORTED "${_production_supported}" PARENT_SCOPE)
  set(GUFO_HIP_EXPERIMENTAL "${_experimental}" PARENT_SCOPE)
endfunction()

function(gufo_require_hip_capability target capability)
  if (capability STREQUAL "production")
    set(_available "${GUFO_HIP_PRODUCTION_SUPPORTED}")
  elseif (capability STREQUAL "wmma-wave32")
    set(_available "${GUFO_HIP_HAS_WMMA_WAVE32}")
  elseif (capability STREQUAL "mfma-wave64")
    set(_available "${GUFO_HIP_HAS_MFMA_WAVE64}")
  else()
    message(FATAL_ERROR "Unknown Gufo HIP capability: ${capability}")
  endif()

  if (NOT _available)
    message(FATAL_ERROR
      "Target ${target} requires HIP capability ${capability}, unavailable on ${GUFO_HIP_ARCHITECTURE}")
  endif()
endfunction()
