#pragma once

#include <cstddef>
#include <cstdint>

#include "local_shm_protocol.h"

namespace recstore {

constexpr std::size_t kLocalShmAlignment = 64;

inline std::size_t
AlignUp(std::size_t value, std::size_t alignment = kLocalShmAlignment) {
  return (value + alignment - 1) / alignment * alignment;
}

inline std::size_t ControlBlockOffset() { return 0; }

inline std::size_t SlotHeadersOffset() {
  return AlignUp(sizeof(LocalShmControlBlock));
}

inline std::size_t SlotHeadersBytes(uint32_t slot_count) {
  return AlignUp(
      sizeof(LocalShmSlotHeader) * static_cast<std::size_t>(slot_count));
}

inline std::size_t SlotPayloadsOffset(uint32_t slot_count) {
  return SlotHeadersOffset() + SlotHeadersBytes(slot_count);
}

inline std::size_t SlotPayloadOffset(
    uint32_t slot_count, uint32_t slot_buffer_bytes, uint32_t slot_id) {
  return SlotPayloadsOffset(slot_count) +
         static_cast<std::size_t>(slot_id) * AlignUp(slot_buffer_bytes);
}

inline std::size_t
TotalRegionBytes(uint32_t slot_count, uint32_t slot_buffer_bytes) {
  return SlotPayloadsOffset(slot_count) +
         static_cast<std::size_t>(slot_count) * AlignUp(slot_buffer_bytes);
}

} // namespace recstore
