#include "ps/local_shm/local_shm_server.h"
#include "ps/local_shm/local_shm_futex.h"

#include <cstring>
#include <thread>
#include <vector>

#include "base/factory.h"
#include "base/log.h"
#include "ps/base/cache_ps_impl.h"

namespace recstore {

namespace {

void FinishWithStatus(LocalShmSlotHeader* header, LocalStatusCode code) {
  header->status_code = static_cast<uint32_t>(code);
  header->state.store(code == LocalStatusCode::kOk
                          ? static_cast<uint32_t>(LocalSlotState::kDone)
                          : static_cast<uint32_t>(LocalSlotState::kError),
                      std::memory_order_release);
  header->completion_doorbell.fetch_add(1, std::memory_order_release);
  FutexWakeAll(&header->completion_doorbell);
}

} // namespace

LocalShmStoreRuntime::LocalShmStoreRuntime(LocalShmRegion* region,
                                           ::CachePS* cache_ps)
    : region_(region), cache_ps_(cache_ps) {}

void LocalShmStoreRuntime::Run() {
  while (!stop_.load()) {
    auto* control = region_->control();
    const uint32_t observed_before_scan =
        control->request_doorbell.load(std::memory_order_acquire);
    bool found_work = false;
    for (uint32_t slot_id = 0; slot_id < region_->slot_count(); ++slot_id) {
      auto* header = region_->slot_header(slot_id);
      uint32_t expected = static_cast<uint32_t>(LocalSlotState::kReady);
      if (header->state.compare_exchange_strong(
              expected, static_cast<uint32_t>(LocalSlotState::kRunning))) {
        ProcessSlot(slot_id);
        found_work = true;
      }
    }
    if (!found_work) {
      if (!stop_.load(std::memory_order_acquire)) {
        FutexWaitUntilValueChange(
            &control->request_doorbell,
            observed_before_scan,
            std::chrono::milliseconds(100));
      }
    }
  }
}

void LocalShmStoreRuntime::Stop() {
  stop_.store(true, std::memory_order_release);
  auto* control = region_->control();
  control->request_doorbell.fetch_add(1, std::memory_order_release);
  FutexWakeAll(&control->request_doorbell);
}

void LocalShmStoreRuntime::ProcessSlot(uint32_t slot_id) {
  auto* header = region_->slot_header(slot_id);
  auto* payload = region_->slot_payload(slot_id);
  try {
    switch (static_cast<LocalOpcode>(header->opcode)) {
      case LocalOpcode::kInitTable: {
        const std::string table_name(reinterpret_cast<const char*>(payload),
                                     header->table_name_len);
        const uint8_t* cursor = payload + header->table_name_len;
        uint64_t num_embeddings = 0;
        uint64_t embedding_dim = 0;
        std::memcpy(&num_embeddings, cursor, sizeof(num_embeddings));
        cursor += sizeof(num_embeddings);
        std::memcpy(&embedding_dim, cursor, sizeof(embedding_dim));
        const bool ok =
            cache_ps_->InitTable(table_name, num_embeddings, embedding_dim);
        FinishWithStatus(
            header, ok ? LocalStatusCode::kOk : LocalStatusCode::kUnknownError);
        return;
      }
      case LocalOpcode::kGet: {
        const auto* keys = reinterpret_cast<const uint64_t*>(payload);
        std::vector<ParameterPack> packs;
        packs.reserve(header->key_count);
        const base::ConstArray<uint64_t> key_array(keys, header->key_count);
        if (!cache_ps_->GetParameterRun2Completion(key_array, packs, 0)) {
          FinishWithStatus(header, LocalStatusCode::kUnknownError);
          return;
        }
        int64_t embedding_dim = 0;
        for (const auto& pack : packs) {
          embedding_dim = std::max<int64_t>(embedding_dim, pack.dim);
        }
        const std::size_t output_bytes =
            sizeof(float) * packs.size() * static_cast<std::size_t>(embedding_dim);
        if (output_bytes > region_->slot_buffer_bytes()) {
          FinishWithStatus(header, LocalStatusCode::kBufferTooSmall);
          return;
        }
        std::memset(payload, 0, output_bytes);
        float* out = reinterpret_cast<float*>(payload);
        for (std::size_t row = 0; row < packs.size(); ++row) {
          if (packs[row].dim > 0 && packs[row].emb_data != nullptr) {
            std::copy_n(packs[row].emb_data,
                        packs[row].dim,
                        out + row * static_cast<std::size_t>(embedding_dim));
          }
        }
        header->embedding_dim = static_cast<uint32_t>(embedding_dim);
        header->output_bytes = output_bytes;
        FinishWithStatus(header, LocalStatusCode::kOk);
        return;
      }
      case LocalOpcode::kPut: {
        const auto* keys = reinterpret_cast<const uint64_t*>(payload);
        const auto* values =
            reinterpret_cast<const float*>(payload +
                                           sizeof(uint64_t) * header->key_count);
        cache_ps_->PutDenseParameterBatch(keys,
                                          values,
                                          static_cast<int>(header->key_count),
                                          static_cast<int>(header->embedding_dim),
                                          0);
        FinishWithStatus(header, LocalStatusCode::kOk);
        return;
      }
      case LocalOpcode::kUpdateFlat: {
        const std::string table_name(reinterpret_cast<const char*>(payload),
                                     header->table_name_len);
        const uint8_t* cursor = payload + header->table_name_len;
        const auto* keys = reinterpret_cast<const uint64_t*>(cursor);
        cursor += sizeof(uint64_t) * header->key_count;
        const auto* grads = reinterpret_cast<const float*>(cursor);
        const base::ConstArray<uint64_t> key_array(keys, header->key_count);
        const bool ok = cache_ps_->UpdateParameterFlat(
            table_name,
            key_array,
            grads,
            static_cast<int64_t>(header->key_count),
            static_cast<int64_t>(header->embedding_dim),
            0);
        FinishWithStatus(
            header, ok ? LocalStatusCode::kOk : LocalStatusCode::kUnknownError);
        return;
      }
      default:
        FinishWithStatus(header, LocalStatusCode::kUnsupportedOpcode);
        return;
    }
  } catch (const std::exception& e) {
    LOG(ERROR) << "LocalShmStoreRuntime::ProcessSlot exception: " << e.what();
    FinishWithStatus(header, LocalStatusCode::kUnknownError);
    return;
  }
}

void LocalShmParameterServer::Init(const json& config) {
  config_ = config;
  local_config_ = config.contains("local_shm") ? config["local_shm"] : json::object();
  const uint32_t slot_count = local_config_.value("slot_count", 64);
  const uint32_t slot_buffer_bytes =
      local_config_.value("slot_buffer_bytes", 8 * 1024 * 1024);
  const std::string region_name =
      local_config_.value("region_name", "recstore_local_ps");

  region_ = std::make_unique<LocalShmRegion>();
  CHECK(region_->Create(region_name, slot_count, slot_buffer_bytes));
  cache_ps_ = std::make_shared<CachePS>(config_["cache_ps"]);
  runtime_ = std::make_unique<LocalShmStoreRuntime>(region_.get(), cache_ps_.get());
}

LocalShmParameterServer::~LocalShmParameterServer() = default;

void LocalShmParameterServer::Run() {
  CHECK(runtime_ != nullptr);
  runtime_->Run();
}

void LocalShmParameterServer::Stop() {
  if (runtime_ != nullptr) {
    runtime_->Stop();
  }
}

FACTORY_REGISTER(BaseParameterServer,
                 LocalShmParameterServer,
                 LocalShmParameterServer);

} // namespace recstore
