#include "report_client.h"
#include <glog/logging.h>
#include <curl/curl.h>
#include <nlohmann/json.hpp>

#include <thread>
#include <mutex>
#include <queue>
#include <condition_variable>
#include <filesystem>
#include <fstream>

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
  AsyncReportQueue::GetInstance().Enqueue(json_payload);
  return true;
}

extern "C" bool
report(const char* table_name,
       const char* unique_id,
       const char* metric_name,
       double metric_value) {
  json j                   = {{"table_name", table_name},
                              {"unique_id", unique_id},
                              {"metric_name", metric_name},
                              {"metric_value", metric_value}};
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