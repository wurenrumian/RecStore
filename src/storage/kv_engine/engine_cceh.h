#pragma once

#include <sys/user.h>

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <memory>
#include <string>

#include "../nvm/pet_kv/shm_common.h"
#include "../ssd/CCEH.h"
#include "base/factory.h"
#include "base_kv.h"
#include "memory/persist_malloc.h"
#include "pair.h"
#include "storage/ssd/io_backend.h"
#include "storage/ssd/io_backend_factory.h"

class KVEngineCCEH : public BaseKV {
  static constexpr int kKVEngineValidFileSize = 123;

public:
  KVEngineCCEH(const BaseKVConfig& config) : BaseKV(config) {
    value_size_      = config.json_config_.at("value_size").get<int>();
    queue_cnt_       = config.json_config_.at("queue_size").get<int>();
    type             = config.json_config_.at("type").get<std::string>();
    pages_per_value_ = (value_size_ + PAGE_SIZE - 1) / PAGE_SIZE;
    LOG(INFO) << "--------------init KVEngineCCEH--------------------";
    std::string index_path = config.json_config_.at("path").get<std::string>();
    std::string index_db_path = index_path + "/cceh_test.db";
    std::string value_path = config.json_config_.at("path").get<std::string>();
    std::string value_db_path = value_path + "/cceh_value.db";
    BackendType backend_type =
        (type == "SPDK") ? BackendType::SPDK : BackendType::IOURING;
    IOConfig io_config_index{backend_type, queue_cnt_, index_db_path, 0};

    hash_table_ = new CCEH(io_config_index);
    // For SPDK, offset value pages to avoid overlapping with index LBAs on the
    // raw device. 1M pages * 4KB = 4GB reserved for index. Adjust if index
    // grows larger.
    PageID_t value_offset = (backend_type == BackendType::SPDK) ? 1000000 : 1;
    IOConfig io_config_value{
        backend_type, queue_cnt_, value_db_path, value_offset};
    value_io_backend = IOBackendFactory::create(io_config_value);
    value_io_backend->init();

    LOG(INFO) << "After init value and  index io_backend ";
  }

  void Get(const uint64_t key, std::string& value, unsigned tid) override {
    Key_t hash_key         = key;
    PageID_t start_page_id = NONE;
    start_page_id          = hash_table_->Get(hash_key);

    if (start_page_id == NONE) {
      value = std::string();
    } else {
      value.resize(value_size_);
      for (uint64_t i = 0; i < pages_per_value_; i++) {
        PageID_t pid  = start_page_id + i;
        char* buffer  = (char*)value_io_backend->GetPage(pid);
        uint64_t size = (i == pages_per_value_ - 1)
                          ? value_size_ - i * PAGE_SIZE
                          : PAGE_SIZE;
        memcpy(value.data() + i * PAGE_SIZE, buffer, size);
        value_io_backend->Unpin(pid, buffer, false);
      }
    }
  }

  void Put(const uint64_t key,
           const std::string_view& value,
           unsigned tid) override {
    PageID_t start_page_id;
    uint64_t written_bytes = 0;
    for (int i = 0; i < pages_per_value_; i++) {
      PageID_t page_id = value_io_backend->AllocatePage();
      if (i == 0)
        start_page_id = page_id;
      char* buffer = (char*)value_io_backend->GetPage(page_id);
      if (written_bytes < value.size()) {
        uint64_t size = std::min(
            (uint64_t)PAGE_SIZE, (uint64_t)value.size() - written_bytes);
        memcpy(buffer, value.data() + written_bytes, size);
        written_bytes += size;
      }
      value_io_backend->Unpin(page_id, buffer, true);
    }
    Key_t hash_key = key;
    hash_table_->Insert(hash_key, start_page_id);
  }

  void BatchGet(base::ConstArray<uint64_t> keys,
                std::vector<base::ConstArray<float>>* values,
                unsigned tid) override {
    values->clear();
    int size = keys.Size();
    std::vector<Value_t> vals(size, NONE);
    pending = 0;
    coros.clear();
    coros.reserve(size);
    for (size_t i = 0; i < size; i++) {
      auto k = keys[i];
      coros.push_back(std::make_unique<coroutine<void>::pull_type>(
          [this, &vals, i, k](auto& yield) {
            vals[i] = hash_table_->Get(yield, i, k);
          }));
    }
    while (pending)
      hash_table_->io_backend->PollCompletion();
    // Collect valid keys and prepare read buffers
    std::vector<IOBackend::IOEntry> read_entries;
    std::vector<int> valid_indices; // which vals[] entries are valid
    read_entries.reserve(size);
    valid_indices.reserve(size);
    for (int i = 0; i < size; i++) {
      if (vals[i] != NONE) {
        char* buffer = value_io_backend->AllocateBuffer(pages_per_value_);
        read_entries.push_back({(PageID_t)vals[i], buffer, pages_per_value_});
        valid_indices.push_back(i);
      }
    }
    // Batch read all values at once
    value_io_backend->BatchReadPages(read_entries);

    // Copy results into output
    static thread_local std::vector<std::vector<float>> value_buffers;
    value_buffers.clear();
    value_buffers.reserve(size);
    int ri = 0; // index into read_entries
    for (int i = 0; i < size; i++) {
      if (ri < (int)valid_indices.size() && valid_indices[ri] == i) {
        value_buffers.emplace_back(value_size_ / sizeof(float));
        auto& buf = value_buffers.back();
        memcpy((char*)buf.data(),
               read_entries[ri].buffer,
               std::min(value_size_, (int)(pages_per_value_ * PAGE_SIZE)));
        value_io_backend->FreeBuffer(read_entries[ri].buffer);
        values->emplace_back(buf.data(), buf.size());
        ri++;
      } else {
        values->emplace_back();
        value_buffers.emplace_back();
      }
    }
  }

  void BatchPut(base::ConstArray<uint64_t> keys,
                std::vector<base::ConstArray<float>>* values,
                unsigned tid) override {
    // Phase 1: allocate page IDs
    std::vector<PageID_t> start_page_ids(keys.Size(), NONE);
    for (int i = 0; i < keys.Size(); i++) {
      start_page_ids[i] = value_io_backend->GetNextPageID();
      value_io_backend->SetNextPageID(start_page_ids[i] + pages_per_value_);
    }

    // Phase 2: prepare per-key contiguous buffers and batch write
    std::vector<IOBackend::IOEntry> write_entries;
    write_entries.reserve(keys.Size());
    for (int j = 0; j < keys.Size(); j++) {
      auto value           = (*values)[j];
      uint64_t value_bytes = value.Size() * sizeof(float);
      // One contiguous buffer for all pages of this key (zero-filled)
      char* buffer = value_io_backend->AllocateBuffer(pages_per_value_);
      uint64_t copy_size =
          std::min(value_bytes, pages_per_value_ * (uint64_t)PAGE_SIZE);
      memcpy(buffer, (const char*)value.Data(), copy_size);
      write_entries.push_back({start_page_ids[j], buffer, pages_per_value_});
    }
    value_io_backend->BatchWritePages(write_entries);
    for (auto& e : write_entries) {
      value_io_backend->FreeBuffer(e.buffer);
    }

    // Phase 3: CCEH index inserts
    for (int j = 0; j < keys.Size(); j++) {
      Key_t hash_key = keys[j];
      hash_table_->Insert(hash_key, start_page_ids[j]);
    }
  }

  ~KVEngineCCEH() {
    std::cout << "exit KVEngineCCEH" << std::endl;
    if (hash_table_) {
      delete hash_table_;
      hash_table_ = nullptr;
    }
  }

private:
  CCEH* hash_table_;
  std::string dict_pool_name_;
  int value_size_;
  int queue_cnt_;
  std::string type;
  uint64_t pages_per_value_;
  std::unique_ptr<IOBackend> value_io_backend;
};

FACTORY_REGISTER(BaseKV, KVEngineCCEH, KVEngineCCEH, const BaseKVConfig&);