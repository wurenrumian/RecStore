#pragma once
#include <memory>
#include <functional>
#include "io_backend.h"

class IOBackendFactory {
public:
  using BackendCreator = std::function<std::unique_ptr<IOBackend>(IOConfig&)>;
  static void register_backend(BackendType type, BackendCreator creator);
  static std::unique_ptr<IOBackend> create(IOConfig& cfg);
};