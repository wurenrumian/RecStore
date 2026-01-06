#include <iostream>
#include <fstream>
#include <string>
#include <algorithm>

#include <gflags/gflags.h>
#include <folly/init/Init.h>

#include "base/base.h"
#include "base/factory.h"
#include "base/json.h"
#include "base_ps/base_ps_server.h"
#include "recstore_config.h"
#include "report_client.h"

DECLARE_string(config_path);
DECLARE_string(brpc_config_path);

using recstore::BaseParameterServer;

static inline std::string ToUpper(std::string s) {
  std::transform(s.begin(), s.end(), s.begin(), [](unsigned char c) {
    return std::toupper(c);
  });
  return s;
}

int main(int argc, char** argv) {
  folly::Init(&argc, &argv);
  xmh::Reporter::StartReportThread(2000);

  std::string cfg_path = FLAGS_config_path;
  {
    std::ifstream test(cfg_path);
    if (!test.good()) {
      std::ifstream test_b(FLAGS_brpc_config_path);
      if (test_b.good())
        cfg_path = FLAGS_brpc_config_path;
    }
  }

  std::ifstream config_file(cfg_path);
  if (!config_file.good()) {
    std::cerr << "Failed to open config file: " << cfg_path << std::endl;
    return 1;
  }

  json config;
  config_file >> config;

  std::string ps_type = "GRPC";
  try {
    if (config.contains("cache_ps") && config["cache_ps"].contains("ps_type")) {
      ps_type = config["cache_ps"]["ps_type"].get<std::string>();
    }
  } catch (...) {
  }

  std::string key;
  std::string type_upper = ToUpper(ps_type);
  if (type_upper == "GRPC") {
    key = "GRPCParameterServer";
  } else if (type_upper == "BRPC") {
    key = "BRPCParameterServer";
  } else {
    std::cerr << "Unknown ps_type: " << ps_type << ", expected GRPC or BRPC"
              << std::endl;
    return 2;
  }

  std::cout << "Using ps_type: " << type_upper << " (key=" << key << ")"
            << std::endl;
  std::cout << "Parameter server config: " << config.dump(2) << std::endl;

  std::unique_ptr<BaseParameterServer> server(
      base::Factory<BaseParameterServer>::NewInstance(key));
  server->Init(config);
  server->Run();

  return 0;
}
