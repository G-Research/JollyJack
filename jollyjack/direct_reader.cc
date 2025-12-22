/**
 * @brief Implements an Arrow RandomAccessFile that performs direct I/O reads
 *        with O_DIRECT mode and configurable block size.
 *
 * @author Alan Fitton
 */

#include "direct_reader.h"

#include <fcntl.h>
#include <unistd.h>
#include <sys/stat.h>
#include <cstring>
#include <stdexcept>
#include <iostream>

#include <arrow/result.h>
#include <arrow/status.h>
#include <arrow/buffer.h>
#include <arrow/util/future.h>

DirectReader::DirectReader(const std::string& filename, size_t block_size)
    : filename_(filename), pos_(0), size_(0), is_closed_(false),
      block_size_(block_size) {
  fd_ = open(filename_.c_str(), O_RDONLY | O_DIRECT);
  if (fd_ < 0) {
    throw std::runtime_error("Failed to open file with O_DIRECT: " + filename_);
  }

  struct stat st;
  if (fstat(fd_, &st) < 0) {
    close(fd_);
    throw std::runtime_error("fstat failed: " + filename_);
  }
  size_ = st.st_size;
}

DirectReader::~DirectReader() {
  (void)Close();
}

arrow::Status DirectReader::Close() {
  if (!is_closed_) {
    close(fd_);
    is_closed_ = true;
  }
  return arrow::Status::OK();
}

bool DirectReader::closed() const {
  return is_closed_;
}

arrow::Status DirectReader::Seek(int64_t position) {
  pos_ = position;
  return arrow::Status::OK();
}

arrow::Result<int64_t> DirectReader::Tell() const {
  return pos_;
}

arrow::Result<int64_t> DirectReader::Read(int64_t nbytes, void* out) {
  ARROW_ASSIGN_OR_RAISE(auto buffer, ReadAt(pos_, nbytes));
  memcpy(out, buffer->data(), buffer->size());
  pos_ += buffer->size();
  return buffer->size();
}

arrow::Result<std::shared_ptr<arrow::Buffer>>
DirectReader::Read(int64_t nbytes) {
  ARROW_ASSIGN_OR_RAISE(auto buffer, ReadAt(pos_, nbytes));
  pos_ += buffer->size();
  return buffer;
}

arrow::Result<std::shared_ptr<arrow::Buffer>>
DirectReader::ReadAt(int64_t position, int64_t nbytes) {
  // Align position and size to block_size_
  int64_t aligned_pos = (position / block_size_) * block_size_;
  int64_t offset = position - aligned_pos;
  int64_t aligned_size =
      ((nbytes + offset + block_size_ - 1) / block_size_) * block_size_;

  // Allocate aligned buffer
  void* aligned_buffer = nullptr;
  if (posix_memalign(&aligned_buffer, block_size_, aligned_size) != 0) {
    return arrow::Status::OutOfMemory("Failed to allocate aligned buffer");
  }

  // Perform direct read
  ssize_t bytes_read = pread(fd_, aligned_buffer, aligned_size, aligned_pos);
  if (bytes_read < 0) {
    free(aligned_buffer);
    return arrow::Status::IOError("pread failed");
  }

  // Copy requested data to Arrow buffer
  ARROW_ASSIGN_OR_RAISE(auto buffer, arrow::AllocateResizableBuffer(nbytes));
  
  int64_t copy_size = std::min(nbytes, bytes_read - offset);
  if (copy_size > 0) {
    memcpy(buffer->mutable_data(),
           static_cast<char*>(aligned_buffer) + offset, copy_size);
  }
  
  free(aligned_buffer);

  if (copy_size < nbytes) {
    ARROW_RETURN_NOT_OK(buffer->Resize(copy_size));
  }

  return std::shared_ptr<arrow::Buffer>(std::move(buffer));
}

arrow::Future<std::shared_ptr<arrow::Buffer>>
DirectReader::ReadAsync(const arrow::io::IOContext& ctx, int64_t position,
                        int64_t nbytes) {
  return RandomAccessFile::ReadAsync(ctx, position, nbytes);
}

arrow::Result<int64_t> DirectReader::GetSize() {
  return size_;
}