/* Win32 shadow of <sys/random.h>; implemented with the system CNG RNG. */
#pragma once
#include "gufo_posix.h"

#ifdef __cplusplus
extern "C" {
#endif

ssize_t getrandom(void* buffer, size_t length, unsigned int flags);

#ifdef __cplusplus
}
#endif
