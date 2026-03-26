#include "io_backend_factory.h"
#include "spdk_backend.h"
#include "io_uring_backend.h"

namespace {
static std::unordered_map<BackendType, IOBackendFactory::BackendCreator>
    registry;
}

void IOBackendFactory::register_backend(BackendType type,
                                        BackendCreator creator) {
  registry[type] = std::move(creator);
}

std::unique_ptr<IOBackend> IOBackendFactory::create(IOConfig& cfg) {
  auto it = registry.find(cfg.type);
  if (it == registry.end())
    throw std::runtime_error("Unknown backend type");
  return it->second(cfg);
}

struct FactoryRegistrar {
  FactoryRegistrar() {
    IOBackendFactory::register_backend(BackendType::SPDK, [](IOConfig& cfg) {
      return std::make_unique<SpdkBackend>(cfg);
    });

    IOBackendFactory::register_backend(BackendType::IOURING, [](IOConfig& cfg) {
      return std::make_unique<IoUringBackend>(cfg);
    });
  }
};

static FactoryRegistrar registrar;