#include <gtest/gtest.h>

#include <string>
#include <vector>

#include "ps/local_shm/local_shm_queue.h"
#include "ps/local_shm/local_shm_region.h"

namespace recstore {
namespace {

std::string UniqueRegionName() {
  return "recstore_local_shm_region_test_" + std::to_string(::getpid());
}

TEST(LocalShmRegionTest, CreateAndAttachRegion) {
  const std::string region_name = UniqueRegionName();
  constexpr uint32_t kSlotCount = 4;
  constexpr uint32_t kSlotBytes = 4096;

  LocalShmRegion server_region;
  ASSERT_TRUE(server_region.Create(region_name, kSlotCount, kSlotBytes));
  ASSERT_TRUE(server_region.IsOpen());
  EXPECT_EQ(server_region.control()->magic, kLocalShmMagic);
  EXPECT_EQ(server_region.control()->slot_count, kSlotCount);
  EXPECT_EQ(server_region.control()->slot_buffer_bytes, kSlotBytes);

  LocalShmRegion client_region;
  ASSERT_TRUE(client_region.Attach(region_name, kSlotCount, kSlotBytes));
  ASSERT_TRUE(client_region.IsOpen());
  EXPECT_EQ(client_region.control()->magic, kLocalShmMagic);
  EXPECT_NE(client_region.slot_header(0), nullptr);
  EXPECT_NE(client_region.slot_payload(0), nullptr);

  std::vector<uint32_t> seen;
  seen.reserve(kSlotCount);
  for (uint32_t i = 0; i < kSlotCount; ++i) {
    uint32_t slot_id = 0;
    ASSERT_TRUE(LocalShmQueueDequeue(client_region.queue_header(LocalQueueKind::kFree),
                                     client_region.queue_cells(LocalQueueKind::kFree),
                                     &slot_id));
    seen.push_back(slot_id);
  }
  EXPECT_FALSE(LocalShmQueueDequeue(client_region.queue_header(LocalQueueKind::kFree),
                                    client_region.queue_cells(LocalQueueKind::kFree),
                                    &seen[0]));
  EXPECT_EQ(seen.size(), static_cast<size_t>(kSlotCount));
}

} // namespace
} // namespace recstore
