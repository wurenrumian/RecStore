#include <folly/init/Init.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <iostream>
#include <memory>
#include <numeric>
#include <string>
#include <vector>

#include "base/array.h"
#include "benchmark/ps_transport_benchmark_config.h"
#include "framework/ps_client_factory.h"
#include "ps/brpc/brpc_ps_client.h"

DEFINE_string(transport, "grpc", "grpc|brpc|rdma|local_shm");
DEFINE_string(host, "127.0.0.1", "server host");
DEFINE_int32(port, 15000, "server port");
DEFINE_int32(num_shards, 1, "number of shards");
DEFINE_int32(iterations, 100, "number of mixed read/update iterations");
DEFINE_int32(rounds, 1, "number of measured rounds");
DEFINE_int32(warmup_rounds, 0, "number of warmup rounds before measurement");
DEFINE_int32(batch_keys, 128, "number of keys per iteration");
DEFINE_int32(embedding_dim, 128, "embedding dimension");
DEFINE_string(report_mode,
              "summary",
              "benchmark output mode: summary|per_round|both");
DEFINE_string(table_name, "default", "embedding table name");
DEFINE_double(update_scale, 0.001, "gradient scale per key");

namespace {

std::string NormalizeMixedTransport(std::string transport) {
  std::transform(
      transport.begin(),
      transport.end(),
      transport.begin(),
      [](unsigned char c) { return static_cast<char>(std::toupper(c)); });
  if (transport == "RDMA" || transport == "GRPC" || transport == "BRPC" ||
      transport == "LOCAL_SHM") {
    return transport;
  }
  throw std::invalid_argument("Unsupported transport: " + transport);
}

nlohmann::json BuildMixedBenchmarkConfig(
    const std::string& transport, const std::string& host, int port) {
  const auto normalized = NormalizeMixedTransport(transport);
  if (normalized == "LOCAL_SHM") {
    return {{"cache_ps", {{"ps_type", "LOCAL_SHM"}}},
            {"client", {{"host", host}, {"port", port}, {"shard", 0}}},
            {"region_name", "recstore_local_ps"}};
  }
  return BuildRpcBenchmarkConfig(normalized, host, port);
}

bool BenchmarkUsesVectorGetForMixed(const std::string& transport) {
  return NormalizeMixedTransport(transport) == "BRPC";
}

bool ShouldPrintPerRound(const std::string& mode) {
  return mode == "per_round" || mode == "both";
}

bool ShouldPrintSummary(const std::string& mode) {
  return mode == "summary" || mode == "both";
}

double PercentileUs(std::vector<int64_t> samples, double ratio) {
  CHECK(!samples.empty());
  CHECK_GE(ratio, 0.0);
  CHECK_LE(ratio, 1.0);
  std::sort(samples.begin(), samples.end());
  const std::size_t idx = static_cast<std::size_t>(std::min<int64_t>(
      samples.size() - 1,
      static_cast<int64_t>(std::ceil(ratio * samples.size()) - 1)));
  return static_cast<double>(samples[idx]);
}

void PrintSummary(const std::string& transport,
                  const std::string& phase,
                  const std::vector<int64_t>& elapsed_us_samples,
                  int iterations_per_round,
                  int batch_keys) {
  if (elapsed_us_samples.empty()) {
    return;
  }

  const double total_us = std::accumulate(
      elapsed_us_samples.begin(), elapsed_us_samples.end(), 0.0);
  const double mean_us =
      total_us / static_cast<double>(elapsed_us_samples.size());
  const double p50_us = PercentileUs(elapsed_us_samples, 0.50);
  const double p95_us = PercentileUs(elapsed_us_samples, 0.95);
  const double p99_us = PercentileUs(elapsed_us_samples, 0.99);

  const double mixed_ops_per_round =
      static_cast<double>(iterations_per_round) * 2.0; // read + update
  const double key_ops_per_round =
      mixed_ops_per_round * static_cast<double>(batch_keys);
  const double total_rounds = static_cast<double>(elapsed_us_samples.size());
  const double ops_per_sec =
      (mixed_ops_per_round * total_rounds) / (total_us / 1e6);
  const double key_ops_per_sec =
      (key_ops_per_round * total_rounds) / (total_us / 1e6);

  std::cout << "system=RecStore transport=" << transport << " phase=" << phase
            << " summary rounds=" << elapsed_us_samples.size()
            << " iterations=" << iterations_per_round
            << " batch_keys=" << batch_keys << " elapsed_us_mean=" << mean_us
            << " elapsed_us_p50=" << p50_us << " elapsed_us_p95=" << p95_us
            << " elapsed_us_p99=" << p99_us << " ops_per_sec=" << ops_per_sec
            << " key_ops_per_sec=" << key_ops_per_sec << std::endl;
}

std::vector<uint64_t> MakeKeys(int batch_keys) {
  CHECK_GT(batch_keys, 0) << "--batch_keys must be positive";
  std::vector<uint64_t> keys;
  keys.reserve(static_cast<std::size_t>(batch_keys));
  for (int i = 0; i < batch_keys; ++i) {
    keys.push_back(static_cast<uint64_t>(1001 + i));
  }
  return keys;
}

std::vector<std::vector<float>>
MakeValues(const std::vector<uint64_t>& keys, int embedding_dim) {
  std::vector<std::vector<float>> values;
  values.reserve(keys.size());
  for (auto key : keys) {
    std::vector<float> row(
        static_cast<std::size_t>(embedding_dim), static_cast<float>(key));
    values.push_back(std::move(row));
  }
  return values;
}

std::vector<float> MakeFlatGradients(
    const std::vector<uint64_t>& keys, int embedding_dim, double update_scale) {
  std::vector<float> grads(
      static_cast<std::size_t>(keys.size()) *
          static_cast<std::size_t>(embedding_dim),
      0.0f);
  for (std::size_t row = 0; row < keys.size(); ++row) {
    const float value = static_cast<float>(keys[row] * update_scale);
    for (int col = 0; col < embedding_dim; ++col) {
      grads[row * static_cast<std::size_t>(embedding_dim) +
            static_cast<std::size_t>(col)] = value;
    }
  }
  return grads;
}

} // namespace

int main(int argc, char** argv) {
  folly::Init(&argc, &argv);

  const std::string transport   = NormalizeMixedTransport(FLAGS_transport);
  const std::string report_mode = FLAGS_report_mode;
  CHECK(report_mode == "summary" || report_mode == "per_round" ||
        report_mode == "both")
      << "Invalid --report_mode: " << report_mode
      << ", expected summary|per_round|both";
  CHECK_GT(FLAGS_embedding_dim, 0) << "--embedding_dim must be positive";

  const auto keys   = MakeKeys(FLAGS_batch_keys);
  const auto values = MakeValues(keys, FLAGS_embedding_dim);
  const auto grads =
      MakeFlatGradients(keys, FLAGS_embedding_dim, FLAGS_update_scale);
  const auto key_array   = base::ConstArray<uint64_t>(keys);
  const int total_rounds = FLAGS_warmup_rounds + FLAGS_rounds;

  auto config = BuildMixedBenchmarkConfig(transport, FLAGS_host, FLAGS_port);
  std::unique_ptr<recstore::BasePSClient> client(
      recstore::CreateFrameworkPSClient(config));

  CHECK_EQ(client->InitEmbeddingTable(
               FLAGS_table_name,
               recstore::EmbeddingTableConfig{
                   static_cast<uint64_t>(std::max(FLAGS_batch_keys, 1024)),
                   static_cast<uint64_t>(FLAGS_embedding_dim)}),
           0)
      << transport << " InitEmbeddingTable failed";
  CHECK_EQ(client->PutParameter(key_array, values), 0)
      << transport << " PutParameter failed during initialization";

  std::vector<int64_t> warmup_samples_us;
  std::vector<int64_t> measure_samples_us;
  warmup_samples_us.reserve(std::max(0, FLAGS_warmup_rounds));
  measure_samples_us.reserve(std::max(0, FLAGS_rounds));

  for (int round = 0; round < total_rounds; ++round) {
    const bool is_warmup = round < FLAGS_warmup_rounds;
    auto start           = std::chrono::steady_clock::now();
    for (int i = 0; i < FLAGS_iterations; ++i) {
      if (BenchmarkUsesVectorGetForMixed(transport)) {
        auto* brpc_client = dynamic_cast<BRPCParameterClient*>(client.get());
        CHECK_NE(brpc_client, nullptr);
        std::vector<std::vector<float>> output;
        CHECK_EQ(brpc_client->GetParameter(key_array, &output), 0)
            << transport << " GetParameter failed at iteration=" << i;
      } else {
        std::vector<float> output(
            keys.size() * static_cast<std::size_t>(FLAGS_embedding_dim), 0.0f);
        CHECK_EQ(client->GetParameter(key_array, output.data()), 0)
            << transport << " GetParameter failed at iteration=" << i;
      }
      CHECK_EQ(client->UpdateParameterFlat(
                   FLAGS_table_name,
                   key_array,
                   grads.data(),
                   static_cast<int64_t>(keys.size()),
                   FLAGS_embedding_dim),
               0)
          << transport << " UpdateParameterFlat failed at iteration=" << i;
    }
    auto end = std::chrono::steady_clock::now();
    const int64_t elapsed_us =
        std::chrono::duration_cast<std::chrono::microseconds>(end - start)
            .count();
    (is_warmup ? warmup_samples_us : measure_samples_us).push_back(elapsed_us);
    if (ShouldPrintPerRound(report_mode)) {
      std::cout << "system=RecStore transport=" << transport
                << " phase=" << (is_warmup ? "warmup" : "measure") << " round="
                << (is_warmup ? (round + 1) : (round - FLAGS_warmup_rounds + 1))
                << "/" << (is_warmup ? FLAGS_warmup_rounds : FLAGS_rounds)
                << " elapsed_us=" << elapsed_us << std::endl;
    }
  }

  if (ShouldPrintSummary(report_mode)) {
    PrintSummary(transport,
                 "warmup",
                 warmup_samples_us,
                 FLAGS_iterations,
                 FLAGS_batch_keys);
    PrintSummary(transport,
                 "measure",
                 measure_samples_us,
                 FLAGS_iterations,
                 FLAGS_batch_keys);
  }
  return 0;
}
