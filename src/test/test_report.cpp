#include <gtest/gtest.h>
#include "base/report/report_client.h"
#include <chrono>
#include <thread>
#include <string>
#include <iostream>

using namespace std::chrono_literals;

TEST(ReportTest, TimelineBasic) {
  {
    ReportTimeline tl("cpp_easy_test_timeline_map", "timeline_test_2");
    std::cout << "Running phase ... " << std::endl;
    std::this_thread::sleep_for(1s);
  }

  std::this_thread::sleep_for(500ms);
  SUCCEED();
}

TEST(ReportTest, FlameGraphData) {
  FlameGraphData root     = {"main", 0.0, 0, 100.0, 10.0};
  FlameGraphData child1   = {"do_work", 10.0, 1, 60.0, 20.0};
  FlameGraphData child2   = {"do_io", 70.0, 1, 30.0, 30.0};
  FlameGraphData child1_1 = {"compute", 20.0, 2, 40.0, 40.0};

  report_flame_graph("cpp_easy_test_flame_map", "flame_test_2", root);
  report_flame_graph("cpp_easy_test_flame_map", "flame_test_2", child1);
  report_flame_graph("cpp_easy_test_flame_map", "flame_test_2", child2);
  report_flame_graph("cpp_easy_test_flame_map", "flame_test_2", child1_1);

  std::this_thread::sleep_for(500ms);
  SUCCEED();
}
