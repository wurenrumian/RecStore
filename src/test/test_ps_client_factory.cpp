#include <gtest/gtest.h>

#include "framework/ps_client_factory.h"

namespace recstore {

TEST(PSClientFactoryTest, RejectsRdmaForFrameworkUsage) {
  json config = {
      {"cache_ps", {{"ps_type", "RDMA"}}},
      {"client", {{"host", "127.0.0.1"}, {"port", 25000}, {"shard", 0}}},
  };

  EXPECT_THROW(
      {
        const auto type = ResolveFrameworkPSType(config);
        (void)type;
      },
      std::invalid_argument);
}

TEST(PSClientFactoryTest, RejectsUnknownType) {
  json config = {{"cache_ps", {{"ps_type", "banana"}}}};

  EXPECT_THROW(
      {
        const auto type = ResolveFrameworkPSType(config);
        (void)type;
      },
      std::invalid_argument);
}

TEST(PSClientFactoryTest, UsesDefaultGrpcClientConfig) {
  json config = {{"cache_ps", {{"ps_type", "grpc"}}}};

  EXPECT_EQ(ResolveFrameworkPSType(config), "GRPC");

  json client_config = ResolveFrameworkClientConfig(config);
  EXPECT_EQ(client_config["host"], "127.0.0.1");
  EXPECT_EQ(client_config["port"], 15000);
  EXPECT_EQ(client_config["shard"], 0);
}

TEST(PSClientFactoryTest, PreservesExplicitBrpcClientConfig) {
  json config = {
      {"cache_ps", {{"ps_type", "BRPC"}}},
      {"client", {{"host", "10.0.0.5"}, {"port", 25123}, {"shard", 1}}},
  };

  EXPECT_EQ(ResolveFrameworkPSType(config), "BRPC");

  json client_config = ResolveFrameworkClientConfig(config);
  EXPECT_EQ(client_config["host"], "10.0.0.5");
  EXPECT_EQ(client_config["port"], 25123);
  EXPECT_EQ(client_config["shard"], 1);
}

} // namespace recstore
