#include "arrow/api.h"
#include "arrow/io/api.h"
#include "arrow/result.h"
#include "arrow/util/logging.h"
#include "arrow/util/type_fwd.h"
#include "parquet/arrow/reader.h"
#include "parquet/arrow/writer.h"
#include "parquet/arrow/schema.h"
#include "parquet/column_reader.h"

#include "palletjack.h"

#include <iostream>
#include <fstream>
#include <chrono>
#include <memory>

using arrow::Status;

void ReadColumnChunk(const parquet::FileMetaData& file_metadata, const char *parquet_path, void* data, int row_group, int column)
{
    std::cerr 
      << " ReadColumnChunk rows" << file_metadata.num_rows()
      << " row_groups:" << file_metadata.num_row_groups() 
      << " columns:" << file_metadata.num_columns()
      << std::endl;

    std::cerr 
      << " row_group:" << row_group
      << " column:" << column
      << std::endl;

    auto arrowReaderProperties = parquet::default_arrow_reader_properties();
    arrowReaderProperties.set_pre_buffer(true);
    parquet::arrow::FileReaderBuilder fileReaderBuilder;
    auto readerProperties = parquet::default_reader_properties();
    fileReaderBuilder.properties(arrowReaderProperties);
    fileReaderBuilder.OpenFile(parquet_path, false, readerProperties);
    auto reader = fileReaderBuilder.Build();
    auto row_reader = reader->get()->RowGroup(row_group);
    auto column_reader = row_reader.get()->Column(column);
    auto float_reader = (parquet::TypedColumnReader<parquet::FloatType>*)(column_reader.get());
    auto row_group_metadata = file_metadata.RowGroup(row_group);
    auto num_rows = row_group_metadata->num_rows();
    /*
    int16_t* def_levels = nullptr;
    int16_t* rep_levels = nullptr;
    int64_t values_read = 0;
    auto read_levels = float_reader->ReadBatch(num_rows, def_levels, rep_levels, (float*)data, &values_read);
    values_read = values_read;
    */
   {}
}