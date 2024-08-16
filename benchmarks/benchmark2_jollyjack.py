
import pyarrow.parquet as pq
import pyarrow as pa
import numpy as np
import torch


tensor = torch.zeros(10, 10).transpose(0, 1)
tensor.numpy()
