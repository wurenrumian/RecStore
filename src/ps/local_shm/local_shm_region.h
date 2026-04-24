#pragma once

#include <cstddef>
#include <cstdint>
#include <string>

#include "base/log.h"
#include "local_shm_layout.h"

namespace recstore {

class LocalShmRegion {
public:
  LocalShmRegion() = default;
  ~LocalShmRegion();

  LocalShmRegion(const LocalShmRegion&)            = delete;
  LocalShmRegion& operator=(const LocalShmRegion&) = delete;

  bool Create(const std::string& region_name,
              uint32_t slot_count,
              uint32_t slot_buffer_bytes);
  bool Attach(const std::string& region_name,
              uint32_t expected_slot_count        = 0,
              uint32_t expected_slot_buffer_bytes = 0);
  void Close();

  bool IsOpen() const { return base_ != nullptr; }
  const std::string& region_name() const { return region_name_; }
  uint32_t slot_count() const { return slot_count_; }
  uint32_t slot_buffer_bytes() const { return slot_buffer_bytes_; }
  std::size_t mapped_size() const { return mapped_size_; }

  LocalShmControlBlock* control();
  const LocalShmControlBlock* control() const;
  LocalShmSlotHeader* slot_header(uint32_t slot_id);
  const LocalShmSlotHeader* slot_header(uint32_t slot_id) const;
  uint8_t* slot_payload(uint32_t slot_id);
  const uint8_t* slot_payload(uint32_t slot_id) const;

private:
  bool MapRegion(int fd, std::size_t mapped_size);
  bool ValidateGeometry(uint32_t expected_slot_count,
                        uint32_t expected_slot_buffer_bytes) const;

private:
  std::string region_name_;
  int fd_                     = -1;
  void* base_                 = nullptr;
  std::size_t mapped_size_    = 0;
  uint32_t slot_count_        = 0;
  uint32_t slot_buffer_bytes_ = 0;
};

} // namespace recstore
