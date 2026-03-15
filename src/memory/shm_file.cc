#include "shm_file.h"

#include <algorithm>
#include <iostream>
#ifdef __linux__
#  include <execinfo.h>
#  include <sys/stat.h>
#endif

namespace base {

bool ShmFile::InitializeFsDax(const std::string& filename, int64 size) {
  ClearFsDax();

  std::error_code ec;
  bool file_exists = fs::exists(filename, ec);
  if (ec) {
    LOG(ERROR) << "fs::exists failed for " << filename << ": " << ec.message();
    return false;
  }

  if (!file_exists) {
    fs::create_directories(fs::path(filename).parent_path(), ec);
    if (ec) {
      LOG(ERROR) << "fs::create_directories failed: " << ec.message();
      return false;
    }
    LOG(INFO) << "Create ShmFile: " << filename << ", size: " << size;
  }

  fd_ = open(filename.c_str(), O_RDWR | O_CREAT, 0666);
  if (fd_ < 0) {
    LOG(ERROR) << "Failed to open file " << filename << ": " << strerror(errno);
    return false;
  }

  if (!file_exists) {
    int ret = posix_fallocate(fd_, 0, size);
    if (ret != 0) {
      LOG(ERROR) << "posix_fallocate failed: " << strerror(ret);
      close(fd_);
      fd_ = -1;
      return false;
    }
  }

  struct stat st;
  if (fstat(fd_, &st) != 0) {
    LOG(ERROR) << "fstat failed: " << strerror(errno);
    close(fd_);
    fd_ = -1;
    return false;
  }
  size_ = st.st_size;

  if (size_ != size) {
    LOG(ERROR) << "Size Error: " << size_ << " vs " << size;
    close(fd_);
    fd_ = -1;
    return false;
  }

  data_ = reinterpret_cast<char*>(
      mmap(nullptr, size_, PROT_READ | PROT_WRITE, MAP_SHARED, fd_, 0));
  CHECK_NE(data_, MAP_FAILED) << "map failed";

  filename_ = filename;
  LOG(INFO) << "mmap shm file: " << filename << ", size: " << size_
            << ", fd: " << fd_;
  return true;
}

bool ShmFile::InitializeDevDax(const std::string& filename, int64 size) {
  {
    static std::mutex m;
    std::lock_guard<std::mutex> _(m);
    data_ =
        (char*)PMMmapRegisterCenter::GetInstance()->Register(filename, size);
    filename_ = filename;
  }

  std::error_code ec;
  if (!fs::exists(filename, ec)) {
    fs::create_directories(fs::path(filename).parent_path(), ec);
    LOG(INFO) << "Create ShmFile: " << filename << ", size: " << size;
    std::ofstream output(filename);
    output.write("a", 1);
    output.close();
  }
  size_ = size;
  return true;
}

bool ShmFile::Initialize(const std::string& filename, int64 size) {
  LOG(INFO) << "[ShmFile::Initialize] type_=" << type_
            << ", filename=" << filename << ", size=" << size;
  std::error_code ec;
  if (!fs::exists("/dev/dax0.0", ec) && type_ == "DRAM") {
    LOG(INFO) << "[ShmFile::Initialize] /dev/dax0.0 not found; override type_ "
                 "DRAM -> SSD";
    type_ = "SSD";
  }
#ifdef __linux__
  void* bt[20];
  int bt_size    = ::backtrace(bt, 20);
  char** bt_syms = ::backtrace_symbols(bt, bt_size);
  LOG(INFO) << "[ShmFile::Initialize] Backtrace:";
  for (int i = 0; i < bt_size; ++i) {
    LOG(INFO) << bt_syms[i];
  }
  free(bt_syms);
#endif
  if (type_ == "DRAM") {
    LOG(INFO) << "ShmFile, devdax mode, type_:" << type_
              << ", filename:" << filename;
    return InitializeDevDax(filename, size);
  } else if (type_ == "SSD" || filename.find("valid") != std::string::npos) {
    LOG(INFO) << "ShmFile, fsdax mode, type_:" << type_
              << ", filename:" << filename;
    return InitializeFsDax(filename, size);
  } else {
    LOG(INFO) << "Unsupport Medium! type_:" << type_
              << ", filename:" << filename;
    return false;
  }
}

void ShmFile::ClearDevDax() {}

void ShmFile::ClearFsDax() {
  if (fd_ >= 0) {
    LOG(INFO) << "ummap shm file: " << filename_ << ", size: " << size_
              << ", fd: " << fd_;
    filename_.clear();
    munmap(data_, size_);
    close(fd_);
    data_ = NULL;
    size_ = 0;
    fd_   = -1;
  }
}
void ShmFile::Clear() {
  std::error_code ec;
  if (fs::exists("/dev/dax0.0", ec)) {
    ClearDevDax();
  } else {
    ClearFsDax();
  }
}

} // namespace base
