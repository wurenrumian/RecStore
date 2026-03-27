#pragma once
#include <string>

extern "C" {

bool report(const char* table_name,
            const char* unique_id,
            const char* metric_name,
            double metric_value);
}

bool send_json_request(const std::string& json_payload);

bool is_report_remote_enabled_for_test();
bool is_report_local_jsonl_enabled_for_test();

namespace recstore {
extern thread_local uint64_t g_trace_id;
}

struct FlameGraphData {
  std::string label;
  double start;
  int level;
  double value;
  double self;
};

extern "C" {
bool report_flame_graph(
    const char* table_name, const char* unique_id, const FlameGraphData& data);
}

enum TimelineState { STATE_RUNNING = 1, STATE_SUCCESS = 2, STATE_FAILED = 3 };

class ReportTimeline {
public:
  ReportTimeline(const std::string& table_name,
                 const std::string& unique_id,
                 const std::string& metric_name = "state");

  ~ReportTimeline();

  void done(TimelineState state = STATE_SUCCESS);

  ReportTimeline(const ReportTimeline&)            = delete;
  ReportTimeline& operator=(const ReportTimeline&) = delete;

private:
  std::string table_name_;
  std::string unique_id_;
  std::string metric_name_;
  bool is_done_;
};
