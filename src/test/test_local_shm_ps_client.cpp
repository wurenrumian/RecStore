#include <gtest/gtest.h>

#include <chrono>
#include <thread>
#include <vector>

#include "base/json.h"
#include "ps/local_shm/local_shm_client.h"
#include "ps/local_shm/local_shm_server.h"

namespace recstore {
namespace {

json MakeLocalShmConfig(const std::string& region_name,
                        uint32_t ready_queue_count = 1) {
  return {
      {"cache_ps",
       {{"num_threads", 1},
        {"ps_type", "LOCAL_SHM"},
        {"base_kv_config",
         {{"path", "/tmp/recstore_local_shm_test"},
          {"index_type", "DRAM"},
          {"value_type", "DRAM"},
          {"capacity", 1024},
          {"value_size", 16}}}}},
      {"local_shm",
       {{"region_name", region_name},
        {"slot_count", 8},
        {"ready_queue_count", ready_queue_count},
        {"slot_buffer_bytes", 1 << 20},
        {"client_timeout_ms", 1000}}},
  };
}

TEST(LocalShmPSClientTest, FactoryClientTypeCanBeConstructed) {
  const auto config = MakeLocalShmConfig(
      "recstore_local_shm_ps_client_factory_" + std::to_string(::getpid()));
  LocalShmParameterServer server;
  server.Init(config);

  std::thread server_thread([&]() { server.Run(); });

  std::this_thread::sleep_for(std::chrono::milliseconds(10));

  LocalShmPSClient client(config["local_shm"]);
  EXPECT_EQ(client.InitEmbeddingTable("table_a", {128, 4}), 0);

  server.Stop();
  server_thread.join();
}

TEST(LocalShmPSClientTest, PutGetAndUpdateFlatRoundTrip) {
  const auto config = MakeLocalShmConfig(
      "recstore_local_shm_ps_client_rw_" + std::to_string(::getpid()));
  LocalShmParameterServer server;
  server.Init(config);

  std::thread server_thread([&]() { server.Run(); });
  std::this_thread::sleep_for(std::chrono::milliseconds(10));

  LocalShmPSClient client(config["local_shm"]);
  ASSERT_EQ(client.InitEmbeddingTable("table_b", {128, 4}), 0);

  std::vector<uint64_t> keys             = {1, 5};
  std::vector<std::vector<float>> values = {
      {1.0f, 2.0f, 3.0f, 4.0f}, {5.0f, 6.0f, 7.0f, 8.0f}};
  base::ConstArray<uint64_t> key_array(keys);
  ASSERT_EQ(client.PutParameter(key_array, values), 0);

  std::vector<float> readback(8, 0.0f);
  ASSERT_EQ(client.GetParameter(key_array, readback.data()), 0);
  EXPECT_EQ(readback[0], 1.0f);
  EXPECT_EQ(readback[1], 2.0f);
  EXPECT_EQ(readback[4], 5.0f);
  EXPECT_EQ(readback[7], 8.0f);

  std::vector<float> grads = {1.0f, 1.0f, 1.0f, 1.0f, 2.0f, 2.0f, 2.0f, 2.0f};
  ASSERT_EQ(
      client.UpdateParameterFlat("table_b", key_array, grads.data(), 2, 4), 0);
  ASSERT_EQ(client.GetParameter(key_array, readback.data()), 0);

  server.Stop();
  server_thread.join();
}

TEST(LocalShmPSClientTest, MultiClientUsesIndependentReadyQueues) {
  const auto config = MakeLocalShmConfig(
      "recstore_local_shm_ps_client_multi_" + std::to_string(::getpid()), 2);
  LocalShmParameterServer server;
  server.Init(config);

  std::thread server_thread([&]() { server.Run(); });
  std::this_thread::sleep_for(std::chrono::milliseconds(10));

  json client0_config                 = config["local_shm"];
  json client1_config                 = config["local_shm"];
  client0_config["ready_queue_index"] = 0;
  client1_config["ready_queue_index"] = 1;

  LocalShmPSClient client0(client0_config);
  LocalShmPSClient client1(client1_config);
  ASSERT_EQ(client0.InitEmbeddingTable("table_c", {128, 4}), 0);

  std::thread worker0([&]() {
    std::vector<uint64_t> keys             = {1, 3};
    std::vector<std::vector<float>> values = {
        {1.0f, 1.0f, 1.0f, 1.0f}, {3.0f, 3.0f, 3.0f, 3.0f}};
    base::ConstArray<uint64_t> key_array(keys);
    EXPECT_EQ(client0.PutParameter(key_array, values), 0);
    std::vector<float> readback(8, 0.0f);
    EXPECT_EQ(client0.GetParameter(key_array, readback.data()), 0);
    EXPECT_FLOAT_EQ(readback[0], 1.0f);
    EXPECT_FLOAT_EQ(readback[4], 3.0f);
  });

  std::thread worker1([&]() {
    std::vector<uint64_t> keys             = {2, 4};
    std::vector<std::vector<float>> values = {
        {2.0f, 2.0f, 2.0f, 2.0f}, {4.0f, 4.0f, 4.0f, 4.0f}};
    base::ConstArray<uint64_t> key_array(keys);
    EXPECT_EQ(client1.PutParameter(key_array, values), 0);
    std::vector<float> readback(8, 0.0f);
    EXPECT_EQ(client1.GetParameter(key_array, readback.data()), 0);
    EXPECT_FLOAT_EQ(readback[0], 2.0f);
    EXPECT_FLOAT_EQ(readback[4], 4.0f);
  });

  worker0.join();
  worker1.join();

  server.Stop();
  server_thread.join();
}

} // namespace
} // namespace recstore
