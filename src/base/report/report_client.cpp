#include "report_client.h"
#include "base/timer.h"
#include <glog/logging.h>
#include <curl/curl.h>
#include <nlohmann/json.hpp>

#include <thread>
#include <mutex>
#include <queue>
#include <condition_variable>
#include <filesystem>
#include <fstream>
#include <cstdlib>
#include <algorithm>

#include <chrono>

using json = nlohmann::json;

namespace recstore {
thread_local uint64_t g_trace_id = 0;
}

static std::string GetApiUrl() {
  std::string default_url = "http://127.0.0.1:8081/report";
  try {
    std::filesystem::path current_path = std::filesystem::current_path();
    while (true) {
      std::filesystem::path config_file = current_path / "recstore_config.json";
      if (std::filesystem::exists(config_file)) {
        std::ifstream ifs(config_file);
        if (ifs.is_open()) {
          nlohmann::json j;
          ifs >> j;
          if (j.contains("report_API")) {
            return j["report_API"];
          }
        }
        break;
      }
      if (current_path == current_path.parent_path()) {
        break;
      }
      current_path = current_path.parent_path();
    }
  } catch (const std::exception& e) {
    LOG(WARNING) << "Error reading recstore_config.json: " << e.what();
  }
  return default_url;
}

namespace {

enum class ReportMode {
  kRemote,
  kLocal,
};

enum class LocalSinkMode {
  kGlog,
  kJsonl,
  kBoth,
};

struct StructuredReportEvent {
  std::string table_name;
  std::string unique_id;
  std::string metric_name;
  double metric_value;
  uint64_t timestamp_us;
  std::string source;
};

std::string ToLower(std::string value) {
  std::transform(
      value.begin(), value.end(), value.begin(), [](unsigned char c) {
        return std::tolower(c);
      });
  return value;
}

bool ParseLocalReportMode(const std::string& value) {
  const std::string normalized = ToLower(value);
  return normalized == "local" || normalized == "off" ||
         normalized == "disable" || normalized == "disabled" ||
         normalized == "false" || normalized == "0";
}

bool ParseRemoteReportMode(const std::string& value) {
  const std::string normalized = ToLower(value);
  return normalized == "grafana" || normalized == "remote" ||
         normalized == "on" || normalized == "true" || normalized == "1";
}

LocalSinkMode ResolveLocalSinkMode() {
  if (const char* env_mode = std::getenv("RECSTORE_REPORT_LOCAL_SINK");
      env_mode != nullptr) {
    const std::string mode = ToLower(env_mode);
    if (mode == "jsonl") {
      return LocalSinkMode::kJsonl;
    }
    if (mode == "both") {
      return LocalSinkMode::kBoth;
    }
  }
  return LocalSinkMode::kGlog;
}

bool IsLocalJsonlEnabled() {
  const auto mode = ResolveLocalSinkMode();
  return mode == LocalSinkMode::kJsonl || mode == LocalSinkMode::kBoth;
}

std::string GetLocalJsonlPath() {
  if (const char* env_path = std::getenv("RECSTORE_REPORT_JSONL_PATH");
      env_path != nullptr) {
    return env_path;
  }
  return "recstore_report_events.jsonl";
}

uint64_t GetTimestampUs() {
  return std::chrono::duration_cast<std::chrono::microseconds>(
             std::chrono::system_clock::now().time_since_epoch())
      .count();
}

json ToJson(const StructuredReportEvent& event) {
  return {{"table_name", event.table_name},
          {"unique_id", event.unique_id},
          {"metric_name", event.metric_name},
          {"metric_value", event.metric_value},
          {"timestamp_us", event.timestamp_us},
          {"source", event.source}};
}

void WriteLocalStructuredEvent(const StructuredReportEvent& event) {
  const json event_json        = ToJson(event);
  const std::string serialized = event_json.dump();
  const auto sink_mode         = ResolveLocalSinkMode();

  if (sink_mode == LocalSinkMode::kGlog || sink_mode == LocalSinkMode::kBoth) {
    LOG(INFO) << "REPORT_LOCAL_EVENT " << serialized;
  }

  if (sink_mode == LocalSinkMode::kJsonl || sink_mode == LocalSinkMode::kBoth) {
    const std::filesystem::path output_path(GetLocalJsonlPath());
    const auto parent = output_path.parent_path();
    if (!parent.empty()) {
      std::filesystem::create_directories(parent);
    }
    std::ofstream ofs(output_path, std::ios::app);
    ofs << serialized << '\n';
  }
}

bool TryRecordLatencyMetricToTimer(const StructuredReportEvent& event) {
  double value_ns = 0.0;
  if (event.metric_name == "duration_ns" || event.metric_name == "latency_ns") {
    value_ns = event.metric_value;
  } else if (event.metric_name == "duration_us" ||
             event.metric_name == "latency_us") {
    value_ns = event.metric_value * 1000.0;
  } else {
    return false;
  }

  xmh::Timer::ManualRecordNs(
      event.table_name + "." + event.metric_name, value_ns);
  return true;
}

ReportMode ResolveReportMode() {
  if (const char* env_mode = std::getenv("RECSTORE_REPORT_MODE");
      env_mode != nullptr) {
    const std::string mode(env_mode);
    if (ParseLocalReportMode(mode)) {
      return ReportMode::kLocal;
    }
    if (ParseRemoteReportMode(mode)) {
      return ReportMode::kRemote;
    }
    LOG(WARNING) << "Unknown RECSTORE_REPORT_MODE=" << mode
                 << ", fallback to report_API based behavior.";
  }

  const std::string api_url = GetApiUrl();
  if (api_url.empty()) {
    return ReportMode::kLocal;
  }

  return ReportMode::kRemote;
}

bool IsRemoteReportEnabled() {
  return ResolveReportMode() == ReportMode::kRemote;
}

} // namespace

size_t write_callback(void* contents, size_t size, size_t nmemb, void* userp) {
  ((std::string*)userp)->append((char*)contents, size * nmemb);
  return size * nmemb;
}

class AsyncReportQueue {
public:
  static AsyncReportQueue& GetInstance() {
    static AsyncReportQueue instance;
    return instance;
  }

  void Enqueue(const std::string& payload) {
    {
      std::lock_guard<std::mutex> lock(mtx_);
      if (queue_.size() >= 10000) {
        queue_.pop();
      }
      queue_.push(payload);
    }
    cv_.notify_one();
  }

private:
  AsyncReportQueue() : stop_(false) {
    curl_global_init(CURL_GLOBAL_DEFAULT);
    worker_ = std::thread(&AsyncReportQueue::WorkerLoop, this);
  }

  ~AsyncReportQueue() {
    {
      std::lock_guard<std::mutex> lock(mtx_);
      stop_ = true;
    }
    cv_.notify_all();
    if (worker_.joinable()) {
      worker_.join();
    }
    curl_global_cleanup();
  }

  void WorkerLoop() {
    CURL* curl = curl_easy_init();
    if (!curl)
      return;

    std::string api_url = GetApiUrl();

    struct curl_slist* headers =
        curl_slist_append(nullptr, "Content-Type: application/json");
    curl_easy_setopt(curl, CURLOPT_URL, api_url.c_str());
    curl_easy_setopt(curl, CURLOPT_HTTPHEADER, headers);

    curl_easy_setopt(curl, CURLOPT_TIMEOUT, 3L);

    std::string response_buffer;
    curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, write_callback);
    curl_easy_setopt(curl, CURLOPT_WRITEDATA, &response_buffer);

    while (true) {
      std::string payload;
      {
        std::unique_lock<std::mutex> lock(mtx_);
        cv_.wait(lock, [this]() { return stop_ || !queue_.empty(); });

        if (stop_ && queue_.empty()) {
          break;
        }

        payload = std::move(queue_.front());
        queue_.pop();
      }

      response_buffer.clear();
      curl_easy_setopt(curl, CURLOPT_POST, 1L);
      curl_easy_setopt(curl, CURLOPT_POSTFIELDS, payload.c_str());
      curl_easy_setopt(curl, CURLOPT_POSTFIELDSIZE, payload.length());

      CURLcode res = curl_easy_perform(curl);
      if (res != CURLE_OK) {
        LOG(ERROR) << "CURL perform failed: " << curl_easy_strerror(res);
      } else {
        long http_code = 0;
        curl_easy_getinfo(curl, CURLINFO_RESPONSE_CODE, &http_code);
        if (http_code >= 200 && http_code < 300) {
          LOG(INFO) << "Report HTTP API success (HTTP Code: " << http_code
                    << ").";
        } else {
          LOG(ERROR) << "API failed (HTTP Code: " << http_code << ").";
        }
        DLOG(INFO) << "Server response: " << response_buffer;
      }
    }

    curl_slist_free_all(headers);
    curl_easy_cleanup(curl);
  }

  std::queue<std::string> queue_;
  std::mutex mtx_;
  std::condition_variable cv_;
  bool stop_;
  std::thread worker_;
};

bool send_json_request(const std::string& json_payload) {
  if (!IsRemoteReportEnabled()) {
    DLOG(INFO) << "REPORT_LIB INFO: Remote reporting disabled; drop payload.";
    return true;
  }
  AsyncReportQueue::GetInstance().Enqueue(json_payload);
  return true;
}

bool is_report_remote_enabled_for_test() { return IsRemoteReportEnabled(); }

bool is_report_local_jsonl_enabled_for_test() { return IsLocalJsonlEnabled(); }

extern "C" bool
report(const char* table_name,
       const char* unique_id,
       const char* metric_name,
       double metric_value) {
  const StructuredReportEvent event = {
      .table_name   = table_name,
      .unique_id    = unique_id,
      .metric_name  = metric_name,
      .metric_value = metric_value,
      .timestamp_us = GetTimestampUs(),
      .source       = "report"};

  WriteLocalStructuredEvent(event);
  TryRecordLatencyMetricToTimer(event);

  const json j             = ToJson(event);
  std::string json_payload = j.dump();

  bool success = send_json_request(json_payload);
  if (success) {
    DLOG(INFO) << "REPORT_LIB INFO: Data for ID [" << unique_id
               << "] enqueued successfully.";
  }

  return success;
}

extern "C" bool report_flame_graph(
    const char* table_name, const char* unique_id, const FlameGraphData& data) {
  std::string combined_uid = std::string(unique_id) + "|" + data.label;

  auto now        = std::chrono::system_clock::now();
  auto now_time_t = std::chrono::system_clock::to_time_t(now);
  struct tm gmt_tm;
  gmtime_r(&now_time_t, &gmt_tm);
  gmt_tm.tm_hour           = 0;
  gmt_tm.tm_min            = 0;
  gmt_tm.tm_sec            = 0;
  auto start_of_day_s      = timegm(&gmt_tm);
  uint64_t start_of_day_us = static_cast<uint64_t>(start_of_day_s) * 1000000;

  double adjusted_start = data.start - static_cast<double>(start_of_day_us);

  report(table_name, combined_uid.c_str(), "level", data.level);
  report(table_name, combined_uid.c_str(), "value", data.value);
  report(table_name, combined_uid.c_str(), "self", data.self);
  report(table_name, combined_uid.c_str(), "start", adjusted_start);

  DLOG(INFO) << "REPORT_LIB INFO: Flame Graph Data for ID [" << unique_id
             << ", Label " << data.label << "] enqueued successfully as '"
             << combined_uid << "'. Adjusted start: " << adjusted_start;

  return true;
}

ReportTimeline::ReportTimeline(const std::string& table_name,
                               const std::string& unique_id,
                               const std::string& metric_name)
    : table_name_(table_name),
      unique_id_(unique_id),
      metric_name_(metric_name),
      is_done_(false) {
  report(table_name_.c_str(),
         unique_id_.c_str(),
         metric_name_.c_str(),
         static_cast<double>(STATE_RUNNING));
}

ReportTimeline::~ReportTimeline() {
  if (!is_done_) {
    done(STATE_SUCCESS);
  }
}

void ReportTimeline::done(TimelineState state) {
  if (!is_done_) {
    report(table_name_.c_str(),
           unique_id_.c_str(),
           metric_name_.c_str(),
           static_cast<double>(state));
    is_done_ = true;
  }
}
