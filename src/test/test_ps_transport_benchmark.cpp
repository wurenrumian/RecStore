#include <gtest/gtest.h>

#include "benchmark/ps_transport_benchmark_config.h"

TEST(PSTransportBenchmarkConfigTest, RejectsUnknownTransport) {
  EXPECT_THROW(
      {
        const auto transport = NormalizeBenchmarkTransport("banana");
        (void)transport;
      },
      std::invalid_argument);
}

TEST(PSTransportBenchmarkConfigTest, BuildsRpcConfig) {
  auto config = BuildRpcBenchmarkConfig("BRPC", "127.0.0.1", 25000);
  EXPECT_EQ(config["cache_ps"]["ps_type"], "BRPC");
  EXPECT_EQ(config["client"]["host"], "127.0.0.1");
  EXPECT_EQ(config["client"]["port"], 25000);
}
