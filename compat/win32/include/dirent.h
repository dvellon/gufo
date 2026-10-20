/* Win32 shadow of <dirent.h>; implemented in compat/win32/posix.cpp. */
#pragma once

#ifdef __cplusplus
extern "C" {
#endif

struct dirent {
  /* NTFS components are at most 255 UTF-16 code units: 1024 bytes holds the
   * corresponding UTF-8 name and its terminator. */
  char d_name[1024];
};

typedef struct gufo_DIR DIR;

DIR* fdopendir(int fd);
struct dirent* readdir(DIR* directory);
int closedir(DIR* directory);

#ifdef __cplusplus
}
#endif
