/**
 * @brief Implements an Arrow RandomAccessFile that performs asynchronous reads through io_uring.
 *        The default ReadAsync implementation utilizes the Arrow thread pool.
 *
 * @author Alan Fitton
 */

#include "io_uring_reader_1.h"

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

// Thread-local ring and init
thread_local io_uring* tls_ring = nullptr;

io_uring& GetThreadLocalRing() {
  if (!tls_ring) {
    tls_ring = new io_uring();
    if (io_uring_queue_init(64, tls_ring, 0) < 0) {
      throw std::runtime_error("Failed to init thread-local io_uring");
    }
  }
  return *tls_ring;
}

IoUringReader1::IoUringReader1(const std::string& filename)
    : filename_(filename), pos_(0), size_(0), is_closed_(false) {
  fd_ = open(filename_.c_str(), O_RDONLY);
  if (fd_ < 0) {
    throw std::runtime_error("Failed to open file: " + filename_);
  }

  struct stat st;
  if (fstat(fd_, &st) < 0) {
    close(fd_);
    throw std::runtime_error("fstat failed: " + filename_);
  }
  size_ = st.st_size;
}

IoUringReader1::~IoUringReader1() {
  (void)Close();
}

arrow::Status IoUringReader1::Close() {
  if (!is_closed_) {
    close(fd_);
    is_closed_ = true;
  }
  return arrow::Status::OK();
}

bool IoUringReader1::closed() const {
  return is_closed_;
}

arrow::Status IoUringReader1::Seek(int64_t position) {
  pos_ = position;
  return arrow::Status::OK();
}

arrow::Result<int64_t> IoUringReader1::Tell() const {
  return pos_;
}

arrow::Result<int64_t> IoUringReader1::Read(int64_t nbytes, void* out) {
  ARROW_ASSIGN_OR_RAISE(auto buffer, ReadAt(pos_, nbytes));
  memcpy(out, buffer->data(), buffer->size());
  pos_ += buffer->size();
  return buffer->size();
}

arrow::Result<std::shared_ptr<arrow::Buffer>> IoUringReader1::Read(int64_t nbytes) {
  ARROW_ASSIGN_OR_RAISE(auto buffer, ReadAt(pos_, nbytes));
  pos_ += buffer->size();
  return buffer;
}

arrow::Result<std::shared_ptr<arrow::Buffer>> IoUringReader1::ReadAt(int64_t position, int64_t nbytes) {
  io_uring& ring = GetThreadLocalRing();

  ARROW_ASSIGN_OR_RAISE(auto buffer, arrow::AllocateResizableBuffer(nbytes));

  struct io_uring_sqe* sqe = io_uring_get_sqe(&ring);
  if (!sqe) {
    return arrow::Status::IOError("get_sqe failed");
  }

  io_uring_prep_read(sqe, fd_, buffer->mutable_data(), nbytes, position);
  sqe->flags |= IOSQE_ASYNC;  // encourage async

  int submit_result = io_uring_submit(&ring);
  if (submit_result < 0) {
    return arrow::Status::IOError("submit failed");
  }

  struct io_uring_cqe* cqe = nullptr;
  int wait_result = io_uring_wait_cqe(&ring, &cqe);
  if (wait_result < 0) {
    return arrow::Status::IOError("wait_cqe failed");
  }

  if (cqe->res < 0) {
    io_uring_cqe_seen(&ring, cqe);
    return arrow::Status::IOError("read failed");
  }

  int64_t bytes_read = cqe->res;
  io_uring_cqe_seen(&ring, cqe);

  if (bytes_read != nbytes) {
    return arrow::Status::IOError("partial read: ", bytes_read);
  }

  return std::shared_ptr<arrow::Buffer>(std::move(buffer));
}

// Default ReadAsync() implementation: simply issue the read on the context's executor
arrow::Future<std::shared_ptr<arrow::Buffer>> IoUringReader1::ReadAsync(const arrow::io::IOContext& ctx,
                                                            int64_t position,
                                                            int64_t nbytes) {
  return RandomAccessFile::ReadAsync(ctx, position, nbytes);
}

arrow::Result<int64_t> IoUringReader1::GetSize() {
  return size_;
}