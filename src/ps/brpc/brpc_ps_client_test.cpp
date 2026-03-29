#include <folly/executors/CPUThreadPoolExecutor.h>
#include <folly/init/Init.h>

#include <random>

#include "base/array.h"
#include "base/factory.h"
#include "base/timer.h"
#include "ps/base/base_client.h"
#include "brpc_ps_client.h"

static bool
check_eq_1d(const std::vector<float>& a, const std::vector<float>& b) {
  std::cout << "a: ";
  for (auto& v : a) {
    std::cout << v << " ";
  }
  std::cout << std::endl;
  std::cout << "b: ";
  for (auto& v : b) {
    std::cout << v << " ";
  }
  std::cout << std::endl;
  if (a.size() < b.size())
    return false;

  for (int i = 0; i < b.size(); i++) {
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

void TestFactoryClient() {
  std::cout << "=== Testing Factory Pattern (bRPC) ===" << std::endl;

  json config = {{"host", "127.0.0.1"}, {"port", 15000}, {"shard", 1}};

  std::unique_ptr<recstore::BasePSClient> client(
      base::Factory<recstore::BasePSClient, json>::NewInstance("brpc", config));

  if (!client) {
    std::cerr << "Failed to create bRPC PS client via factory!" << std::endl;
    return;
  }

  std::cout << "Successfully created bRPC PS client via factory" << std::endl;

  auto brpc_client = dynamic_cast<BRPCParameterClient*>(client.get());
  if (brpc_client) {
    // brpc_client->ClearPS();
    // assert empty
    std::vector<uint64_t> keys = {1, 2, 3};
    std::vector<std::vector<float>> emptyvalues(keys.size());
    std::vector<std::vector<float>> rightvalues = {{1}, {2, 2}, {3, 3, 3}};
    std::vector<std::vector<float>> values;

    // insert something
    brpc_client->PutParameter(keys, rightvalues);
    std::cout << "put parameter done" << std::endl;
    // read those
    brpc_client->GetParameter(keys, &values);
    CHECK(check_eq_2d(values, rightvalues));

    // clear all
    brpc_client->ClearPS();
    // read those
    brpc_client->GetParameter(keys, &values);
    CHECK(check_eq_2d(values, emptyvalues));

    std::cout << "load fake data" << std::endl;
    brpc_client->LoadFakeData(100);
    std::cout << "load fake data done" << std::endl;
    std::cout << "dump fake data" << std::endl;
    brpc_client->DumpFakeData(100);
    std::cout << "dump fake data done" << std::endl;

    std::cout << "All bRPC operations passed!" << std::endl;
  }
}

void TestDirectClient() {
  std::cout << "\n=== Testing Direct bRPC Client Creation ===" << std::endl;

  BRPCParameterClient client("127.0.0.1", 15000, 1);

  client.ClearPS();
  // assert empty
  std::vector<uint64_t> keys = {1, 2, 3};
  std::vector<std::vector<float>> emptyvalues(keys.size());
  std::vector<std::vector<float>> rightvalues = {{1}, {2, 2}, {3, 3, 3}};
  std::vector<std::vector<float>> values;

  base::ConstArray<uint64_t> keys_array(keys);
  client.GetParameter(keys_array, &values);
  CHECK(check_eq_2d(values, emptyvalues));

  // insert something
  client.PutParameter(keys, rightvalues);
  // read those
  client.GetParameter(keys_array, &values);
  CHECK(check_eq_2d(values, rightvalues));

  // clear all
  client.ClearPS();
  // read those
  client.GetParameter(keys_array, &values);
  CHECK(check_eq_2d(values, emptyvalues));

  std::cout << "All direct bRPC client operations passed!" << std::endl;
}

void TestPrefetch() {
  std::cout << "\n=== Testing bRPC Prefetch ===" << std::endl;

  BRPCParameterClient client("127.0.0.1", 15000, 1);

  client.ClearPS();

  // 准备测试数据
  std::vector<uint64_t> keys             = {100, 101, 102, 103, 104};
  std::vector<std::vector<float>> values = {
      {1.0f, 2.0f}, {3.0f, 4.0f}, {5.0f, 6.0f}, {7.0f, 8.0f}, {9.0f, 10.0f}};

  // 先写入数据
  client.PutParameter(keys, values);

  // 测试 prefetch
  base::ConstArray<uint64_t> keys_array(keys);
  uint64_t prefetch_id = client.PrefetchParameter(keys_array);

  if (prefetch_id != 0) {
    client.WaitForPrefetch(prefetch_id);

    std::vector<std::vector<float>> fetched_values;
    if (client.GetPrefetchResult(prefetch_id, &fetched_values)) {
      CHECK(check_eq_2d(fetched_values, values));
      std::cout << "Prefetch test passed!" << std::endl;
    } else {
      std::cout << "Failed to get prefetch result" << std::endl;
    }
  } else {
    std::cout << "Prefetch not supported or failed" << std::endl;
  }

  client.ClearPS();
}

int main(int argc, char** argv) {
  folly::Init(&argc, &argv);
  xmh::Reporter::StartReportThread(2000);

  std::cout << "=== bRPC 参数服务器客户端测试 ===" << std::endl;
  std::cout << std::endl;

  try {
    TestFactoryClient();
    TestDirectClient();
    TestPrefetch();

    std::cout << "\n所有 bRPC 测试通过！" << std::endl;
  } catch (const std::exception& e) {
    std::cerr << "Test failed with exception: " << e.what() << std::endl;
    return 1;
  }

  return 0;
}
