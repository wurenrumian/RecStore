#include <gtest/gtest.h>

#include "ps/local_shm/local_shm_layout.h"

namespace recstore {
namespace {

TEST(LocalShmProtocolTest, ControlBlockAndSlotHeaderAreDefined) {
  EXPECT_GT(sizeof(LocalShmControlBlock), 0U);
  EXPECT_GT(sizeof(LocalShmSlotHeader), 0U);
  EXPECT_EQ(alignof(LocalShmControlBlock), 64U);
  EXPECT_EQ(alignof(LocalShmSlotHeader), 64U);
}

TEST(LocalShmProtocolTest, LayoutOffsetsAreMonotonic) {
  constexpr uint32_t kSlotCount = 8;
  constexpr uint32_t kSlotBytes = 4096;

  EXPECT_EQ(ControlBlockOffset(), 0U);
  EXPECT_GE(SlotHeadersOffset(), sizeof(LocalShmControlBlock));
  EXPECT_GT(SlotPayloadsOffset(kSlotCount), SlotHeadersOffset());
  EXPECT_GT(TotalRegionBytes(kSlotCount, kSlotBytes),
            SlotPayloadsOffset(kSlotCount));
  EXPECT_EQ(SlotPayloadOffset(kSlotCount, kSlotBytes, 0),
            SlotPayloadsOffset(kSlotCount));
  EXPECT_GT(SlotPayloadOffset(kSlotCount, kSlotBytes, 1),
            SlotPayloadOffset(kSlotCount, kSlotBytes, 0));
}

} // namespace
} // namespace recstore
