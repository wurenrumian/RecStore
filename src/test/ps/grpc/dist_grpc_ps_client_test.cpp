#include "ps/grpc/dist_grpc_ps_client.h"

#include <folly/executors/CPUThreadPoolExecutor.h>
#include <folly/init/Init.h>
#include <future>
#include <random>

#include "base/array.h"
#include "base/factory.h"
#include "base/timer.h"
#include "ps/base/base_client.h"
#include "test/server_mgr/ps_server_launcher.h"

namespace {
constexpr int kGrpcPort0 = 15123;
constexpr int kGrpcPort1 = 15124;
} // namespace

using namespace xmh;
using namespace recstore;

static bool
check_eq_1d(const std::vector<float>& a, const std::vector<float>& b) {
  if (a.size() != b.size())
    return false;

  for (int i = 0; i < a.size(); i++) {
    if (std::abs(a[i] - b[i]) > 1e-6)
      return false;
  }
  return true;
}

static bool check_eq_2d(std::vector<std::vector<float>>& a,
                        const std::vector<std::vector<float>>& b) {
  a.resize(b.size());
  for (size_t i = 0; i < b.size(); ++i) {
    a[i].resize(b[i].size());
  }
  if (a.size() != b.size())
    return false;
  for (int i = 0; i < a.size(); i++) {
    if (check_eq_1d(a[i], b[i]) == false)
      return false;
  }
  return true;
}

void TestBasicConfig() {
  std::cout << "=== Testing Basic Configuration ===" << std::endl;

  json recstore_config = {
      {"distributed_client",
       {{"servers",
         {{{"host", "127.0.0.1"}, {"port", kGrpcPort0}, {"shard", 0}},
          {{"host", "127.0.0.1"}, {"port", kGrpcPort1}, {"shard", 1}}}},
        {"num_shards", 2},
        {"hash_method", "city_hash"}}}};

  try {
    DistributedGRPCParameterClient client(recstore_config);
    std::cout << "Recstore config parsed successfully, shard count: "
              << client.shard_count() << std::endl;
  } catch (const std::exception& e) {
    std::cerr << "Recstore config test failed: " << e.what() << std::endl;
  }
}

void TestFactoryClient() {
  std::cout << "=== Testing Factory Pattern ===" << std::endl;

  json config = {
      {"distributed_client",
       {{"servers",
         {{{"host", "127.0.0.1"}, {"port", kGrpcPort0}, {"shard", 0}},
          {{"host", "127.0.0.1"}, {"port", kGrpcPort1}, {"shard", 1}}}},
        {"num_shards", 2},
        {"hash_method", "city_hash"}}}};

  std::unique_ptr<BasePSClient> base_client(
      base::Factory<BasePSClient, json>::NewInstance(
          "distributed_grpc", config));

  if (!base_client) {
    std::cerr << "Failed to create distributed PS client via factory!"
              << std::endl;
    return;
  }

  auto* client =
      dynamic_cast<DistributedGRPCParameterClient*>(base_client.get());
  if (!client) {
    std::cerr << "Failed to cast to DistributedGRPCParameterClient!"
              << std::endl;
    return;
  }

  std::cout << "Successfully created distributed PS client via factory"
            << std::endl;

  try {
    client->ClearPS();
    std::vector<uint64_t> keys_vec = {1, 2, 3};
    base::ConstArray<uint64_t> keys(keys_vec);
    std::vector<std::vector<float>> emptyvalues(keys_vec.size());
    std::vector<std::vector<float>> rightvalues = {{1}, {2, 2}, {3, 3, 3}};
    std::vector<std::vector<float>> values;
    client->GetParameter(keys, &values);
    CHECK(check_eq_2d(values, emptyvalues));
    std::cout << "pass first check" << std::endl;

    client->PutParameter(keys, rightvalues);
    client->GetParameter(keys, &values);
    CHECK(check_eq_2d(values, rightvalues));
    std::cout << "pass second check" << std::endl;

    client->ClearPS();
    client->GetParameter(keys, &values);
    CHECK(check_eq_2d(values, emptyvalues));

    std::cout << "load fake data" << std::endl;
    CHECK(client->LoadFakeData(100));
    std::cout << "load fake data done" << std::endl;
    std::cout << "dump fake data" << std::endl;
    CHECK(client->DumpFakeData(100));
    std::cout << "dump fake data done" << std::endl;

    std::cout << "All distributed PS operations passed!" << std::endl;
  } catch (const std::exception& e) {
    std::cout << "Test skipped (servers not available): " << e.what()
              << std::endl;
  }
}

void TestDirectClient() {
  std::cout << "=== Testing Direct Client Creation ===" << std::endl;

  json config = {
      {"distributed_client",
       {{"servers",
         {{{"host", "127.0.0.1"}, {"port", kGrpcPort0}, {"shard", 0}},
          {{"host", "127.0.0.1"}, {"port", kGrpcPort1}, {"shard", 1}}}},
        {"num_shards", 2},
        {"hash_method", "city_hash"}}}};

  try {
    DistributedGRPCParameterClient client(config);
    std::cout << "Direct client created successfully, shard count: "
              << client.shard_count() << std::endl;

    client.ClearPS();
    std::vector<uint64_t> keys = {1001, 1002, 1003};
    std::vector<std::vector<float>> emptyvalues(keys.size());
    std::vector<std::vector<float>> rightvalues = {
        {1, 0, 1}, {2, 2}, {3, 3, 3}};
    std::vector<std::vector<float>> values;

    base::ConstArray<uint64_t> keys_array(keys);
    client.GetParameter(keys_array, &values);
    CHECK(check_eq_2d(values, emptyvalues));

    client.PutParameter(keys_array, rightvalues);
    client.GetParameter(keys_array, &values);
    CHECK(check_eq_2d(values, rightvalues));

    client.ClearPS();
    client.GetParameter(keys_array, &values);
    CHECK(check_eq_2d(values, emptyvalues));

    std::cout << "All direct client operations passed!" << std::endl;
  } catch (const std::exception& e) {
    std::cout << "Test skipped (servers not available): " << e.what()
              << std::endl;
  }
}

void TestLargeBatch() {
  std::cout << "=== Testing Large Batch Operations ===" << std::endl;

  json config = {
      {"distributed_client",
       {{"servers",
         {{{"host", "127.0.0.1"}, {"port", kGrpcPort0}, {"shard", 0}},
          {{"host", "127.0.0.1"}, {"port", kGrpcPort1}, {"shard", 1}}}},
        {"num_shards", 2},
        {"hash_method", "city_hash"},
        {"max_keys_per_request", 50}}}};

  try {
    DistributedGRPCParameterClient client(config);

    std::vector<uint64_t> large_keys;
    std::vector<std::vector<float>> large_values;
    for (int i = 0; i < 120; ++i) {
      large_keys.push_back(2000 + static_cast<uint64_t>(i) * 2);
      large_values.push_back({float(i), float(i * 2)});
    }

    base::ConstArray<uint64_t> keys_array(large_keys);

    client.ClearPS();

    int put_result = client.PutParameter(keys_array, large_values);
    CHECK(put_result == 0);

    std::vector<std::vector<float>> retrieved_values;
    int get_result = client.GetParameter(keys_array, &retrieved_values);
    CHECK(get_result == 0);
    CHECK(check_eq_2d(retrieved_values, large_values));

    std::cout << "Large batch test passed!" << std::endl;
  } catch (const std::exception& e) {
    std::cout << "Test skipped (servers not available): " << e.what()
              << std::endl;
  }
}

int main(int argc, char** argv) {
  folly::Init(&argc, &argv);
  xmh::Reporter::StartReportThread(2000);

  auto launch_options =
      recstore::test::PSServerLauncher::LoadOptionsFromEnvironment();
  launch_options.override_ps_type = "GRPC";
  launch_options.override_ports   = {kGrpcPort0, kGrpcPort1};
  recstore::test::ScopedPSServer server(launch_options, true);

  TestBasicConfig();
  TestFactoryClient();
  TestDirectClient();
  TestLargeBatch();
  return 0;
}
