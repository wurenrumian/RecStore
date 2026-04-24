#include <gtest/gtest.h>

#include "ps/local_shm/local_shm_layout.h"

namespace recstore {
namespace {

TEST(LocalShmProtocolTest, ControlBlockAndSlotHeaderAreDefined) {
  EXPECT_GT(sizeof(LocalShmControlBlock), 0U);
  EXPECT_GT(sizeof(LocalShmQueueHeader), 0U);
  EXPECT_GT(sizeof(LocalShmQueueCell), 0U);
  EXPECT_GT(sizeof(LocalShmSlotHeader), 0U);
  EXPECT_EQ(alignof(LocalShmControlBlock), 64U);
  EXPECT_EQ(alignof(LocalShmQueueHeader), 64U);
  EXPECT_EQ(alignof(LocalShmSlotHeader), 64U);
}

TEST(LocalShmProtocolTest, LayoutOffsetsAreMonotonic) {
  constexpr uint32_t kSlotCount = 8;
  constexpr uint32_t kSlotBytes = 4096;

  EXPECT_EQ(ControlBlockOffset(), 0U);
  EXPECT_GE(QueueHeadersOffset(), sizeof(LocalShmControlBlock));
  EXPECT_GT(QueueCellsOffset(), QueueHeadersOffset());
  EXPECT_GT(SlotHeadersOffset(kSlotCount), QueueCellsOffset());
  EXPECT_GT(SlotPayloadsOffset(kSlotCount), SlotHeadersOffset(kSlotCount));
  EXPECT_GT(TotalRegionBytes(kSlotCount, kSlotBytes),
            SlotPayloadsOffset(kSlotCount));
  EXPECT_EQ(SlotPayloadOffset(kSlotCount, kSlotBytes, 0),
            SlotPayloadsOffset(kSlotCount));
  EXPECT_GT(SlotPayloadOffset(kSlotCount, kSlotBytes, 1),
            SlotPayloadOffset(kSlotCount, kSlotBytes, 0));
}

} // namespace
} // namespace recstore
