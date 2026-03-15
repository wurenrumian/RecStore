#pragma once

#include <boost/coroutine2/all.hpp>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <folly/Format.h>
#include <folly/GLog.h>
#include <folly/Likely.h>
#include <glog/logging.h>
#include <sys/user.h>
#include <vector>

typedef uint64_t PageID_t;
const PageID_t INVALID_PAGE = -1;
using boost::coroutines2::coroutine;
extern thread_local int pending;
extern thread_local std::vector<std::unique_ptr<coroutine<void>::pull_type>>
    coros;

enum class BackendType { SPDK, IOURING };

struct IOConfig {
  BackendType type;
  int queue_cnt;
  std::string file_path;
  IOConfig(BackendType t           = BackendType::SPDK,
           int q                   = 512,
           const std::string& path = "/tmp/test.db")
      : type(t), queue_cnt(q), file_path(path) {}
};

class IOBackend {
public:
  IOBackend(IOConfig& config) : queue_cnt(config.queue_cnt) {}
  virtual ~IOBackend() {}
  virtual void init() = 0;

  PageID_t AllocatePage(coroutine<void>::push_type& sink, uint64_t index) {
    PageID_t new_page_id = next_page_id++;
    WritePageAsync(sink, index, new_page_id, empty_page);
    return new_page_id;
  }
  PageID_t AllocatePage() {
    PageID_t new_page_id = next_page_id++;
    WritePageSync(new_page_id, empty_page);
    return new_page_id;
  }

  void ReadPage(coroutine<void>::push_type& sink,
                uint64_t index,
                PageID_t page_id,
                char* buffer) {
    ReadPageAsync(sink, index, page_id, buffer);
  }
  void ReadPage(PageID_t page_id, char* buffer) {
    ReadPageSync(page_id, buffer);
  }

  void WritePage(coroutine<void>::push_type& sink,
                 uint64_t index,
                 PageID_t page_id,
                 char* buffer) {
    WritePageAsync(sink, index, page_id, buffer);
  }
  void WritePage(PageID_t page_id, char* buffer) {
    WritePageSync(page_id, buffer);
  }

  virtual void* GetPage(
      coroutine<void>::push_type& sink, uint64_t index, PageID_t page_id) = 0;
  virtual void* GetPage(PageID_t page_id)                                 = 0;

  // Unpin a page, if dirty, write it back
  virtual void
  Unpin(coroutine<void>::push_type& sink,
        uint64_t index,
        PageID_t page_id,
        void* page_data,
        bool is_dirty)                                                 = 0;
  virtual void Unpin(PageID_t page_id, void* page_data, bool is_dirty) = 0;

  virtual void PollCompletion() = 0;

protected:
  PageID_t next_page_id;
  char* empty_page = nullptr;
  int queue_cnt;

  virtual void ReadPageAsync(coroutine<void>::push_type& sink,
                             uint64_t index,
                             PageID_t page_id,
                             char* buffer)                  = 0;
  virtual void ReadPageSync(PageID_t page_id, char* buffer) = 0;

  virtual void WritePageAsync(coroutine<void>::push_type& sink,
                              uint64_t index,
                              PageID_t page_id,
                              char* buffer)                  = 0;
  virtual void WritePageSync(PageID_t page_id, char* buffer) = 0;
};