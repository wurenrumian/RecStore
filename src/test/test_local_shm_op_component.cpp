#include <gtest/gtest.h>

#include <chrono>
#include <thread>
#include <unistd.h>
#include <vector>

#include "base/json.h"
#include "base/tensor.h"
#include "framework/common/local_shm_op_component.h"
#include "ps/base/base_client.h"
#include "ps/local_shm/local_shm_client.h"
#define private public
#include "ps/local_shm/local_shm_server.h"
#undef private

namespace {

json MakeLocalShmConfig(const std::string& region_name) {
  return {
      {"cache_ps",
       {{"num_threads", 1},
        {"ps_type", "LOCAL_SHM"},
        {"base_kv_config",
         {{"path", "/tmp/recstore_local_shm_component_test"},
          {"index_type", "DRAM"},
          {"value_type", "DRAM"},
          {"capacity", 1024},
          {"value_size", 16}}}}},
      {"local_shm",
       {{"region_name", region_name},
        {"slot_count", 8},
        {"ready_queue_count", 1},
        {"ready_queue_burst_limit", 8},
        {"slot_buffer_bytes", 1 << 20},
        {"client_timeout_ms", 1000}}},
  };
}

TEST(LocalShmOpComponentTest, LookupFlatReadsValuesFromLocalShmClient) {
  const auto config = MakeLocalShmConfig(
      "recstore_local_shm_component_" + std::to_string(::getpid()));
  recstore::LocalShmParameterServer server;
  server.Init(config);

  std::thread server_thread([&]() { server.Run(); });
  std::this_thread::sleep_for(std::chrono::milliseconds(10));

  recstore::LocalShmPSClient client(config["local_shm"]);
  ASSERT_EQ(client.InitEmbeddingTable("table_component", {128, 4}), 0);

  std::vector<uint64_t> key_storage             = {3, 8};
  std::vector<std::vector<float>> value_storage = {
      {3.0f, 4.0f, 5.0f, 6.0f}, {8.0f, 9.0f, 10.0f, 11.0f}};
  base::ConstArray<uint64_t> key_array(key_storage);
  ASSERT_EQ(client.PutParameter(key_array, value_storage), 0);

  std::vector<float> output_storage(8, 0.0f);
  base::RecTensor key_tensor(key_storage.data(), {2});
  base::RecTensor value_tensor(output_storage.data(), {2, 4});

  recstore::LocalShmLookupFlat(&client, "local_shm", key_tensor, value_tensor);

  EXPECT_FLOAT_EQ(output_storage[0], 3.0f);
  EXPECT_FLOAT_EQ(output_storage[1], 4.0f);
  EXPECT_FLOAT_EQ(output_storage[4], 8.0f);
  EXPECT_FLOAT_EQ(output_storage[7], 11.0f);

  server.Stop();
  server_thread.join();
}

} // namespace
