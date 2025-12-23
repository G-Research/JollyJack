/**
 * @brief Implements an Arrow RandomAccessFile that performs direct I/O reads
 *        with O_DIRECT mode and configurable block size.
 */

#include "direct_reader.h"

#include <fcntl.h>
#include <unistd.h>
#include <sys/stat.h>
#include <iostream>

#include <arrow/result.h>
#include <arrow/status.h>
#include <arrow/buffer.h>
#include <arrow/io/memory.h>
#include <arrow/util/future.h>

DirectReader::DirectReader(const std::string& filename, size_t block_size)
    : pos_(0), size_(0), is_closed_(false),
      block_size_(block_size) {
  fd_ = open(filename.c_str(), O_RDONLY | O_DIRECT);
  if (fd_ < 0) {
    throw std::runtime_error("Failed to open file with O_DIRECT: " + filename);
  }

  struct stat st;
  if (fstat(fd_, &st) < 0) {
    close(fd_);
    throw std::runtime_error("fstat failed: " + filename);
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

arrow::Result<std::shared_ptr<arrow::Buffer>> DirectReader::Read(int64_t nbytes) {
  ARROW_ASSIGN_OR_RAISE(auto buffer, ReadAt(pos_, nbytes));
  pos_ += buffer->size();
  return buffer;
}

arrow::Result<std::shared_ptr<arrow::Buffer>> DirectReader::ReadAt(int64_t position, int64_t nbytes) {
  // Calculate aligned offset and length using bit hacks
  int64_t align_down_mask = ~(block_size_ - 1);
  int64_t aligned_start_offset = position & align_down_mask;
  int64_t original_end_offset = position + nbytes;
  int64_t aligned_end_offset = (original_end_offset + block_size_ - 1) & align_down_mask;
  int64_t aligned_read_length = aligned_end_offset - aligned_start_offset;
  
  ARROW_ASSIGN_OR_RAISE(auto buffer, arrow::AllocateBuffer(aligned_read_length, block_size_));
  ssize_t bytes_read = pread(fd_, buffer->mutable_data(), aligned_read_length, aligned_start_offset);

  if (bytes_read < 0) {
    return arrow::Status::IOError("pread failed");
  }

  return arrow::SliceBuffer(std::move(buffer), position - aligned_start_offset, nbytes);
}

arrow::Future<std::shared_ptr<arrow::Buffer>> 
DirectReader::ReadAsync(const arrow::io::IOContext& ctx, int64_t position, int64_t nbytes) {
  return RandomAccessFile::ReadAsync(ctx, position, nbytes);
}

arrow::Result<int64_t> DirectReader::GetSize() {
  return size_;
}

arrow::Status DirectReader::WillNeed(const std::vector<arrow::io::ReadRange>& ranges)
{
  return arrow::Status::OK();
}
