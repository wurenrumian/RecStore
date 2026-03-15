#include "io_backend.h"
#include <fcntl.h>
#include <liburing.h>

class IoUringBackend : public IOBackend {
private:
  std::string file_path;
  int fd;

  struct ThreadQpair {
    struct io_uring qpair;
    bool initialized = false;
    ThreadQpair()    = default;
    ~ThreadQpair() { release(); }

    static ThreadQpair& instance() {
      static thread_local ThreadQpair thread_qpair;
      return thread_qpair;
    }

    struct io_uring* get(int queue_size) {
      if (!initialized) {
        int ret = io_uring_queue_init(queue_size, &qpair, 0);
        if (ret < 0)
          throw std::runtime_error(
              "Failed to initialize thread-local io_uring: " +
              std::string(strerror(-ret)));
        initialized = true;
      }
      return &qpair;
    }

    void release() {
      if (initialized) {
        io_uring_queue_exit(&qpair);
        initialized = false;
      }
    }
  };
  struct io_uring* get_thread_ring() {
    ThreadQpair& thread_qpair = ThreadQpair::instance();
    return thread_qpair.get(queue_cnt);
  }

  void ReadPageAsync(coroutine<void>::push_type& sink,
                     uint64_t index,
                     PageID_t page_id,
                     char* buffer) override {
    struct io_uring* ring    = get_thread_ring();
    struct io_uring_sqe* sqe = io_uring_get_sqe(ring);
    CHECK_NE(sqe, nullptr) << "Failed to get SQE for read operation";
    pending++;
    io_uring_prep_read(sqe, fd, buffer, PAGE_SIZE, page_id * PAGE_SIZE);
    sqe->user_data = index;
    int ret        = io_uring_submit(ring);
    CHECK_GE(ret, 0) << "Failed to submit read operation: " +
                            std::string(strerror(-ret));
    sink();
  }
  void ReadPageSync(PageID_t page_id, char* buffer) override {
    struct io_uring* ring    = get_thread_ring();
    struct io_uring_sqe* sqe = io_uring_get_sqe(ring);
    CHECK_NE(sqe, nullptr) << "Failed to get SQE for sync read operation";
    io_uring_prep_read(sqe, fd, buffer, PAGE_SIZE, page_id * PAGE_SIZE);
    int ret = io_uring_submit_and_wait(ring, 1);
    CHECK_GE(ret, 0) << "Failed to submit sync read operation: " +
                            std::string(strerror(-ret));
    struct io_uring_cqe* cqe = nullptr;
    ret                      = io_uring_wait_cqe(ring, &cqe);
    CHECK_GE(ret, 0) << "Failed to wait CQE for sync read: " +
                            std::string(strerror(-ret));
    CHECK_NE(cqe, nullptr);
    CHECK_GE(cqe->res, 0) << "Sync read failed: " +
                                 std::string(strerror(-cqe->res));
    io_uring_cqe_seen(ring, cqe);
  }

  void WritePageAsync(coroutine<void>::push_type& sink,
                      uint64_t index,
                      PageID_t page_id,
                      char* buffer) override {
    struct io_uring* ring    = get_thread_ring();
    struct io_uring_sqe* sqe = io_uring_get_sqe(ring);
    CHECK_NE(sqe, nullptr) << "Failed to get SQE for write operation";
    pending++;
    io_uring_prep_write(sqe, fd, buffer, PAGE_SIZE, page_id * PAGE_SIZE);
    sqe->user_data = index;
    int ret        = io_uring_submit(ring);
    CHECK_GE(ret, 0) << "Failed to submit write operation: " +
                            std::string(strerror(-ret));
    sink();
  }
  void WritePageSync(PageID_t page_id, char* buffer) override {
    struct io_uring* ring    = get_thread_ring();
    struct io_uring_sqe* sqe = io_uring_get_sqe(ring);
    CHECK_NE(sqe, nullptr) << "Failed to get SQE for sync write operation";
    io_uring_prep_write(sqe, fd, buffer, PAGE_SIZE, page_id * PAGE_SIZE);
    int ret = io_uring_submit_and_wait(ring, 1);
    CHECK_GE(ret, 0) << "Failed to submit sync write operation: " +
                            std::string(strerror(-ret));
    struct io_uring_cqe* cqe = nullptr;
    ret                      = io_uring_wait_cqe(ring, &cqe);
    CHECK_GE(ret, 0) << "Failed to wait CQE for sync write: " +
                            std::string(strerror(-ret));
    CHECK_NE(cqe, nullptr);
    CHECK_GE(cqe->res, 0) << "Sync write failed: " +
                                 std::string(strerror(-cqe->res));
    io_uring_cqe_seen(ring, cqe);
  }

public:
  IoUringBackend(IOConfig& config)
      : IOBackend(config), file_path(config.file_path) {}
  ~IoUringBackend() {
    CHECK(fd >= 0) << "Invalid file descriptor";
    close(fd);
  }

  void init() override {
    bool exists = (access(file_path.c_str(), F_OK) != -1);
    fd          = open(file_path.c_str(), O_RDWR | O_CREAT, S_IRUSR | S_IWUSR);
    if (fd < 0)
      throw std::runtime_error("Failed to open file: " + file_path);
    if (exists) {
      struct stat file_stat;
      fstat(fd, &file_stat);
      next_page_id = file_stat.st_size / PAGE_SIZE;
    } else {
      next_page_id = 0;
    }
    empty_page = new char[PAGE_SIZE]();
  }

  void* GetPage(coroutine<void>::push_type& sink,
                uint64_t index,
                PageID_t page_id) override {
    char* buffer = new char[PAGE_SIZE];
    ReadPageAsync(sink, index, page_id, buffer);
    return buffer;
  }
  void* GetPage(PageID_t page_id) override {
    char* buffer = new char[PAGE_SIZE];
    ReadPageSync(page_id, buffer);
    return buffer;
  }

  void Unpin(coroutine<void>::push_type& sink,
             uint64_t index,
             PageID_t page_id,
             void* page_data,
             bool is_dirty) override {
    if (is_dirty)
      WritePage(sink, index, page_id, reinterpret_cast<char*>(page_data));
    delete[] reinterpret_cast<char*>(page_data);
  }

  void Unpin(PageID_t page_id, void* page_data, bool is_dirty) override {
    if (is_dirty)
      WritePageSync(page_id, reinterpret_cast<char*>(page_data));
    delete[] reinterpret_cast<char*>(page_data);
  }

  void PollCompletion() override {
    struct io_uring* ring = get_thread_ring();
    struct io_uring_cqe* cqe;
    int ret = io_uring_peek_cqe(ring, &cqe);
    if (ret == -EAGAIN)
      return; // No completion events
    else if (ret < 0)
      throw std::runtime_error(
          "Failed to peek CQE: " + std::string(strerror(-ret)));
    int id = cqe->user_data;
    pending--;
    io_uring_cqe_seen(ring, cqe);
    if (coros[id] && *coros[id])
      (*coros[id])();
    return;
  }
};