#include <arrow/io/interfaces.h>
#include <liburing.h>
#include <string>
#include <memory>
#include <thread>
#include <mutex>
#include <queue>
#include <condition_variable>

class ARROW_EXPORT IoUringRandomAccessFile : public arrow::io::RandomAccessFile {
 public:
  static arrow::Result<std::shared_ptr<IoUringRandomAccessFile>> Open(
      const std::string& path, int queue_depth = 256);

  ~IoUringRandomAccessFile() override;

  // RandomAccessFile interface
  arrow::Result<int64_t> GetSize() override;
  arrow::Result<int64_t> ReadAt(int64_t position, int64_t nbytes, void* out) override;
  arrow::Result<std::shared_ptr<arrow::Buffer>> ReadAt(int64_t position, int64_t nbytes) override;
  
  // Truly asynchronous methods
  arrow::Future<std::shared_ptr<arrow::Buffer>> ReadAsync(const arrow::io::IOContext& ctx, 
                                                         int64_t position, 
                                                         int64_t nbytes) override;
  
  std::vector<arrow::Future<std::shared_ptr<arrow::Buffer>>> ReadManyAsync(
      const arrow::io::IOContext& ctx, 
      const std::vector<arrow::io::ReadRange>& ranges) override;

  // InputStream interface (required by base class)
  arrow::Status Close() override;
  bool closed() const override;
  arrow::Result<int64_t> Tell() const override;
  arrow::Result<int64_t> Read(int64_t nbytes, void* out) override;
  arrow::Result<std::shared_ptr<arrow::Buffer>> Read(int64_t nbytes) override;

  // Seekable interface
  arrow::Status Seek(int64_t position) override;

 private:
  struct AsyncReadRequest;
  
  IoUringRandomAccessFile(int fd, int64_t file_size, int queue_depth);
  
  arrow::Status Initialize();
  void ProcessCompletions();
  void SubmitPendingRequests();
  uint64_t GetNextRequestId();
  
  int fd_;
  int64_t file_size_;
  int64_t position_;
  std::atomic<bool> closed_;
  
  // io_uring state
  struct io_uring ring_;
  int queue_depth_;
  
  // Async request management
  std::atomic<uint64_t> next_request_id_;
  std::mutex requests_mutex_;
  std::unordered_map<uint64_t, std::unique_ptr<AsyncReadRequest>> pending_requests_;
  std::queue<std::unique_ptr<AsyncReadRequest>> submission_queue_;
  std::condition_variable submission_cv_;
  
  // Background thread for processing completions
  std::thread completion_thread_;
  std::atomic<bool> should_stop_;
  arrow::Status completion_thread_status;
};
