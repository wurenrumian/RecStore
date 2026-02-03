#include "report_client.h" // 包含头文件中定义的 report 函数 extern "C" 声明
#include <iostream>
#include <curl/curl.h>
// 假设 nlohmann/json.hpp 已添加到您的项目或系统头文件路径
#include <nlohmann/json.hpp>

// ====================================================================
// 配置
// ====================================================================

using json = nlohmann::json;

// FastAPI 服务地址，请确保服务正在这个地址运行
const std::string API_URL = "http://127.0.0.1:8080/report";

size_t write_callback(void* contents, size_t size, size_t nmemb, void* userp) {
  // 将接收到的内容追加到用户提供的 std::string 缓冲区中
  ((std::string*)userp)->append((char*)contents, size * nmemb);
  return size * nmemb;
}

extern "C" bool
report(const char* table_name,
       const char* unique_id,
       const char* metric_name,
       double metric_value) {
  CURL* curl;
  CURLcode res;
  std::string response_buffer;
  long http_code = 0;
  bool success   = false;

  // 1. 构造 JSON Payload
  json j                   = {{"table_name", table_name},
                              {"unique_id", unique_id},
                              {"metric_name", metric_name},
                              {"metric_value", metric_value}};
  std::string json_payload = j.dump();

  // 2. 初始化 libcurl
  curl_global_init(CURL_GLOBAL_DEFAULT);
  curl = curl_easy_init();
  if (!curl) {
    std::cerr << "REPORT_LIB ERROR: CURL initialization failed." << std::endl;
    return false;
  }

  // 3. 设置 libcurl 选项
  curl_easy_setopt(curl, CURLOPT_URL, API_URL.c_str());
  curl_easy_setopt(curl, CURLOPT_POST, 1L);
  curl_easy_setopt(curl, CURLOPT_POSTFIELDS, json_payload.c_str());
  curl_easy_setopt(curl, CURLOPT_POSTFIELDSIZE, json_payload.length());

  // 设置 Content-Type: application/json
  struct curl_slist* headers = nullptr;
  headers = curl_slist_append(headers, "Content-Type: application/json");
  curl_easy_setopt(curl, CURLOPT_HTTPHEADER, headers);

  // 设置响应接收回调和缓冲区
  curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, write_callback);
  curl_easy_setopt(curl, CURLOPT_WRITEDATA, &response_buffer);

  // 4. 执行请求
  res = curl_easy_perform(curl);

  // 5. 检查结果
  if (res != CURLE_OK) {
    std::cerr << "REPORT_LIB ERROR: CURL perform failed: "
              << curl_easy_strerror(res) << std::endl;
  } else {
    // 获取 HTTP 状态码
    curl_easy_getinfo(curl, CURLINFO_RESPONSE_CODE, &http_code);

    // 判断是否成功 (2xx 状态码)
    if (http_code >= 200 && http_code < 300) {
      std::cout << "REPORT_LIB INFO: Data for ID [" << unique_id
                << "] reported successfully." << std::endl;
      success = true;
    } else {
      std::cerr << "REPORT_LIB ERROR: API failed (HTTP Code: " << http_code
                << ")." << std::endl;
      std::cerr << "REPORT_LIB ERROR: Server response: " << response_buffer
                << std::endl;
    }
  }

  // 6. 清理
  curl_slist_free_all(headers);
  curl_easy_cleanup(curl);
  curl_global_cleanup();

  return success;
}