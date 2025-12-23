
#include <arrow/io/interfaces.h>
#include <liburing.h>
#include <string>
#include <memory>

class DirectReader : public arrow::io::RandomAccessFile {
 public:
  explicit DirectReader(const std::string& filename, size_t block_size);
  ~DirectReader() override;

  arrow::Status Close() override;
  bool closed() const override;
  arrow::Status Seek(int64_t position) override;
  arrow::Result<int64_t> Tell() const override;
  arrow::Result<int64_t> Read(int64_t nbytes, void* out) override;
  arrow::Result<std::shared_ptr<arrow::Buffer>> Read(int64_t nbytes) override;
  arrow::Result<std::shared_ptr<arrow::Buffer>> ReadAt(int64_t position, int64_t nbytes) override;
  arrow::Future<std::shared_ptr<arrow::Buffer>> ReadAsync(const arrow::io::IOContext& ctx, int64_t position,int64_t nbytes) override;
  arrow::Result<int64_t> GetSize() override;
  arrow::Status WillNeed(const std::vector<arrow::io::ReadRange>& ranges);

 private:
  int fd_;
  int64_t pos_ = 0;
  int64_t size_ = 0;
  size_t block_size_;
  bool is_closed_ = false;
};