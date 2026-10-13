#include <cstddef>

#include "QnnModelPal.hpp"

#include <cstdlib>
#include <cstring>
#include <dlfcn.h>
#include <string>

namespace {
thread_local std::string last_error;
}

void *qnn_wrapper_api::dlSym(void *handle, const char *symbol) {
  dlerror();
  void *address = dlsym(handle == DL_DEFAULT ? RTLD_DEFAULT : handle, symbol);
  if (const char *error = dlerror()) {
    last_error = error;
    return nullptr;
  }
  last_error.clear();
  return address;
}

char *qnn_wrapper_api::dlError() {
  return const_cast<char *>(last_error.c_str());
}

char *qnn_wrapper_api::strnDup(const char *source, size_t max_length) {
  if (!source) return nullptr;
  const size_t length = strnlen(source, max_length);
  auto *destination = static_cast<char *>(std::malloc(length + 1));
  if (!destination) return nullptr;
  std::memcpy(destination, source, length);
  destination[length] = '\0';
  return destination;
}
