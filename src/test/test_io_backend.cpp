#include <algorithm>
#include <chrono>
#include <cctype>
#include <cstring>
#include <cstdlib>
#include <filesystem>
#include <memory>
#include <string>
#include <vector>

#include <gtest/gtest.h>

#include "base/factory.h"
#include "storage/io_backend/io_backend.h"
#include "storage/io_backend/io_backend_register.h"

namespace {
BaseKVConfig
MakeConfig(const std::string& backend, const std::string& file_path) {
  BaseKVConfig config;
  config.num_threads_ = 1;
  config.json_config_ = {
      {"io_backend_type", backend},
      {"queue_cnt", 64},
      {"page_id_offset", 1},
      {"file_path", file_path},
  };
  return config;
}

class IOBackendTest : public ::testing::Test {
protected:
  void SetUp() override {
    backend_ = "SPDK";

    const auto ts =
        std::chrono::duration_cast<std::chrono::nanoseconds>(
            std::chrono::steady_clock::now().time_since_epoch())
            .count();
    file_path_ = "/tmp/test_io_backend_" + std::to_string(ts) + ".db";

    BaseKVConfig config = MakeConfig(backend_, file_path_);
    using IOF           = base::Factory<IOBackend, const BaseKVConfig&>;
    backend_impl_.reset(IOF::NewInstance(backend_, config));
    ASSERT_NE(backend_impl_, nullptr);
    backend_impl_->init();
  }

  void TearDown() override {
    backend_impl_.reset();
    if (!file_path_.empty())
      std::filesystem::remove(file_path_);
  }

  std::string backend_;
  std::string file_path_;
  std::unique_ptr<IOBackend> backend_impl_;
};

TEST_F(IOBackendTest, SyncReadWriteRoundTrip) {
  PageID_t page_id = backend_impl_->AllocatePage();

  auto* write_page = static_cast<char*>(backend_impl_->GetPage(page_id));
  ASSERT_NE(write_page, nullptr);
  for (size_t i = 0; i < PAGE_SIZE; ++i) {
    write_page[i] = static_cast<char>((i * 7) % 251);
  }
  backend_impl_->Unpin(page_id, write_page, true);

  auto* read_page = static_cast<char*>(backend_impl_->GetPage(page_id));
  ASSERT_NE(read_page, nullptr);
  for (size_t i = 0; i < PAGE_SIZE; ++i) {
    ASSERT_EQ(read_page[i], static_cast<char>((i * 7) % 251))
        << "mismatch at offset " << i;
  }
  backend_impl_->Unpin(page_id, read_page, false);
}

TEST_F(IOBackendTest, BatchReadWriteRoundTrip) {
  constexpr int kNumPages = 8;

  std::vector<IOBackend::IOEntry> write_entries;
  std::vector<char*> write_buffers;
  write_entries.reserve(kNumPages);
  write_buffers.reserve(kNumPages);

  for (int i = 0; i < kNumPages; ++i) {
    PageID_t page_id = backend_impl_->AllocatePage();
    char* buf        = backend_impl_->AllocateBuffer(1);
    ASSERT_NE(buf, nullptr);
    std::memset(buf, i + 11, PAGE_SIZE);
    write_entries.push_back({page_id, buf, 1});
    write_buffers.push_back(buf);
  }
  backend_impl_->BatchWritePages(write_entries);

  std::vector<IOBackend::IOEntry> read_entries;
  std::vector<char*> read_buffers;
  read_entries.reserve(kNumPages);
  read_buffers.reserve(kNumPages);
  for (int i = 0; i < kNumPages; ++i) {
    char* buf = backend_impl_->AllocateBuffer(1);
    ASSERT_NE(buf, nullptr);
    read_entries.push_back({write_entries[i].page_id, buf, 1});
    read_buffers.push_back(buf);
  }
  backend_impl_->BatchReadPages(read_entries);

  for (int i = 0; i < kNumPages; ++i) {
    for (size_t off = 0; off < PAGE_SIZE; ++off) {
      ASSERT_EQ(read_buffers[i][off], static_cast<char>(i + 11))
          << "page=" << i << ", offset=" << off;
    }
  }

  for (char* buf : write_buffers)
    backend_impl_->FreeBuffer(buf);
  for (char* buf : read_buffers)
    backend_impl_->FreeBuffer(buf);
}

} // namespace
