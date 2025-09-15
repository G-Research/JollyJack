#include "io_uring_reader_2.h"
#include <fcntl.h>
#include <unistd.h>
#include <sys/stat.h>
#include <cstring>
#include <stdexcept>
#include <iostream>
#include <thread>

#include <arrow/result.h>
#include <arrow/status.h>
#include <arrow/buffer.h>
#include <arrow/util/future.h>

struct IoUringReader2::AsyncReadRequest {
  uint64_t id;
  int64_t position;
  int64_t nbytes;
  std::shared_ptr<arrow::ResizableBuffer> buffer;
  arrow::Future<std::shared_ptr<arrow::Buffer>> future;
  
  AsyncReadRequest(uint64_t req_id, int64_t pos, int64_t size)
      : id(req_id), position(pos), nbytes(size) {
    future = arrow::Future<std::shared_ptr<arrow::Buffer>>::Make();
  }
  
  arrow::Future<std::shared_ptr<arrow::Buffer>> GetFuture() {
    return future;
  }
};

arrow::Result<std::shared_ptr<IoUringReader2>> IoUringReader2::Open(const std::string& path, int queue_depth) {
  int fd = open(path.c_str(), O_RDONLY);
  if (fd < 0) {
    return arrow::Status::IOError("Failed to open file: ", path, " - ", strerror(errno));
  }

  struct stat st;
  if (fstat(fd, &st) < 0) {
    close(fd);
    return arrow::Status::IOError("Failed to stat file: ", path, " - ", strerror(errno));
  }

  auto file = std::shared_ptr<IoUringReader2>(
      new IoUringReader2(fd, st.st_size, queue_depth));
  
  ARROW_RETURN_NOT_OK(file->Initialize());
  
  return file;
}

IoUringReader2::IoUringReader2(int fd, int64_t file_size, int queue_depth)
    : fd_(fd), 
      file_size_(file_size), 
      position_(0),
      closed_(false),
      ring_{},
      queue_depth_(queue_depth),
      next_request_id_(1),
      should_stop_(false),
      submission_cv_{},
      requests_mutex_{},
      submission_queue_{},
      completion_thread_{},
      completion_thread_status(arrow::Status::OK()) {
}

IoUringReader2::~IoUringReader2() {

  if (!closed_.load()) {
    Close().ok(); // Ignore errors in destructor
  }
}

arrow::Status IoUringReader2::Initialize() {
  // Initialize io_uring
  int ret = io_uring_queue_init(queue_depth_, &ring_, 0);
  if (ret < 0) {
    return arrow::Status::IOError("Failed to initialize io_uring: ", strerror(-ret));
  }

  // Start completion processing thread
  should_stop_.store(false);
  completion_thread_ = std::thread(&IoUringReader2::ProcessCompletions, this);
  
  return arrow::Status::OK();
}

void IoUringReader2::ProcessCompletions() {

  struct __kernel_timespec no_timeout {};

  while (!should_stop_.load()) {

    // Wait & submit any pending requests
    SubmitPendingRequests();

    struct io_uring_cqe* cqe;
    int ret = io_uring_wait_cqe_timeout(&ring_, &cqe, &no_timeout);
    
    if (ret == -ETIME || ret == -EINTR) {
      continue;
    }
    
    if (ret < 0) {
      completion_thread_status = arrow::Status::IOError("Failed to wait io_uring: ", strerror(-ret));
      break;
    }

    // Process completion
    uint64_t request_id = reinterpret_cast<uint64_t>(io_uring_cqe_get_data(cqe));
    
    std::unique_ptr<AsyncReadRequest> request;
    {
      std::lock_guard<std::mutex> lock(requests_mutex_);
      auto it = pending_requests_.find(request_id);
      if (it == pending_requests_.end())
      {        
        completion_thread_status = arrow::Status::KeyError("Failed to find request id:", request_id);
        break;
      }

      request = std::move(it->second);
      pending_requests_.erase(it);
    }

    if (request) {
      if (cqe->res < 0) {
        // Error occurred
        arrow::Status error = arrow::Status::IOError("Read failed: ", strerror(-cqe->res));
        request->future.MarkFinished(error);
      } else {
        // Success - resize buffer to actual bytes read and complete future
        ARROW_UNUSED(request->buffer->Resize(cqe->res));
        request->future.MarkFinished(std::static_pointer_cast<arrow::Buffer>(request->buffer));
      }
      
      io_uring_cqe_seen(&ring_, cqe);
    }
  }

  std::lock_guard<std::mutex> lock(requests_mutex_);
  for (auto& [id, request] : pending_requests_)
  {
    request->future.MarkFinished(completion_thread_status);
  }
}

void IoUringReader2::SubmitPendingRequests() {
  
  std::unique_lock<std::mutex> lk(requests_mutex_);
  submission_cv_.wait_for (lk, std::chrono::milliseconds(1));
  while (!submission_queue_.empty()) {
    struct io_uring_sqe* sqe = io_uring_get_sqe(&ring_);
    if (!sqe) {
      break; // Queue full, try again later
    }

    auto request = std::move(submission_queue_.front());
    submission_queue_.pop();

    // Prepare read operation
    io_uring_prep_read(sqe, fd_, request->buffer->mutable_data(), 
                       request->nbytes, request->position);
    io_uring_sqe_set_data(sqe, reinterpret_cast<void*>(request->id));

    // Store request for completion handling
    pending_requests_[request->id] = std::move(request);
  }

  // Submit all prepared operations
  if (!pending_requests_.empty()) {
    io_uring_submit(&ring_);
  }
}

uint64_t IoUringReader2::GetNextRequestId() {
  return next_request_id_.fetch_add(1);
}

arrow::Result<int64_t> IoUringReader2::GetSize() {
  if (closed_.load()) {
    return arrow::Status::Invalid("File is closed");
  }
  return file_size_;
}

arrow::Future<std::shared_ptr<arrow::Buffer>> IoUringReader2::ReadAsync(
    const arrow::io::IOContext& ctx, int64_t position, int64_t nbytes) {

  if (closed_.load()) {
    return arrow::Future<std::shared_ptr<arrow::Buffer>>::MakeFinished(
        arrow::Status::Invalid("File is closed"));
  }

  if (position < 0 || nbytes < 0) {
    return arrow::Future<std::shared_ptr<arrow::Buffer>>::MakeFinished(
        arrow::Status::Invalid("Invalid read parameters"));
  }

  if (position >= file_size_) {
    ARROW_ASSIGN_OR_RAISE(auto buffer, arrow::AllocateBuffer(0));
    return arrow::Future<std::shared_ptr<arrow::Buffer>>::MakeFinished(std::move(buffer));
  }

  // Clamp read size to file bounds
  nbytes = std::min(nbytes, file_size_ - position);

  // Create async request
  uint64_t request_id = GetNextRequestId();
  auto request = std::make_unique<AsyncReadRequest>(request_id, position, nbytes);

  // Allocate buffer
  auto buffer_result = arrow::AllocateResizableBuffer(nbytes);
  if (!buffer_result.ok()) {
    return arrow::Future<std::shared_ptr<arrow::Buffer>>::MakeFinished(buffer_result.status());
  }
  request->buffer = std::move(buffer_result).ValueOrDie();

  // Get the future from the promise
  auto future = request->GetFuture();

  // Queue for submission
  {
    std::lock_guard<std::mutex> lock(requests_mutex_);
    submission_queue_.push(std::move(request));
  }

  submission_cv_.notify_one();
  return future;
}

std::vector<arrow::Future<std::shared_ptr<arrow::Buffer>>> 
IoUringReader2::ReadManyAsync(const arrow::io::IOContext& ctx, 
                                      const std::vector<arrow::io::ReadRange>& ranges) {
  std::vector<arrow::Future<std::shared_ptr<arrow::Buffer>>> futures;
  futures.reserve(ranges.size());

  for (const auto& range : ranges) {
    futures.push_back(ReadAsync(ctx, range.offset, range.length));
  }

  return futures;
}

// Synchronous fallback implementations
arrow::Result<int64_t> IoUringReader2::ReadAt(int64_t position, int64_t nbytes, void* out) {
  auto future = ReadAsync(arrow::io::IOContext{}, position, nbytes);
  ARROW_ASSIGN_OR_RAISE(auto buffer, future.result());

  int64_t bytes_read = buffer->size();
  std::memcpy(out, buffer->data(), bytes_read);
  return bytes_read;
}

arrow::Result<std::shared_ptr<arrow::Buffer>> IoUringReader2::ReadAt(int64_t position, int64_t nbytes) {
  auto future = ReadAsync(arrow::io::IOContext{}, position, nbytes);
  return future.result();
}

arrow::Status IoUringReader2::Close() {

  if (closed_.exchange(true)) {
    return arrow::Status::OK(); // Already closed
  }

  // Signal completion thread to stop
  should_stop_.store(true);

  // Wait for completion thread
  if (completion_thread_.joinable()) {
    completion_thread_.join();
  }

  // Clean up io_uring
  if (ring_.ring_fd != 0)
    io_uring_queue_exit(&ring_);

  // Close file descriptor
  if (fd_ >= 0) {
    close(fd_);
    fd_ = -1;
  }

  if (!completion_thread_status.ok())
  {
    return completion_thread_status;
  }

  return arrow::Status::OK();
}

bool IoUringReader2::closed() const {
  return closed_.load();
}

// InputStream interface implementations
arrow::Result<int64_t> IoUringReader2::Tell() const {
  if (closed_.load()) {
    return arrow::Status::Invalid("File is closed");
  }

  return position_;
}

arrow::Result<int64_t> IoUringReader2::Read(int64_t nbytes, void* out) {
  ARROW_ASSIGN_OR_RAISE(int64_t bytes_read, ReadAt(position_, nbytes, out));
  position_ += bytes_read;
  return bytes_read;
}

arrow::Result<std::shared_ptr<arrow::Buffer>> IoUringReader2::Read(int64_t nbytes) {
  ARROW_ASSIGN_OR_RAISE(auto buffer, ReadAt(position_, nbytes));
  position_ += buffer->size();
  return buffer;
}

arrow::Status IoUringReader2::Seek(int64_t position) {
  if (closed_.load()) {
    return arrow::Status::Invalid("File is closed");
  }
  if (position < 0) {
    return arrow::Status::Invalid("Invalid seek position");
  }
  position_ = position;
  return arrow::Status::OK();
}
