import mmap
import os
import tempfile
import unittest

import jollyjack as jj
import numpy as np
import palletjack as pj
import pyarrow as pa
import pyarrow.parquet as pq

chunk_size = 3
n_row_groups = 2
n_rows = n_row_groups * chunk_size


def write_parquet(path, column_names):
    data = np.random.uniform(-100, 100, size=(n_rows, len(column_names))).astype(
        np.float32
    )
    table = pa.Table.from_arrays(
        [pa.array(data[:, i]) for i in range(len(column_names))],
        schema=pa.schema([(name, pa.float32()) for name in column_names]),
    )
    pq.write_table(
        table, path, row_group_size=chunk_size, use_dictionary=False, store_schema=False
    )
    index_path = path + ".index"
    pj.generate_metadata_index(path, index_path)
    return index_path


def read_task_by_task(
    np_array, sources, index_paths, column_sets, source_column_set, tasks
):
    """The per-task Python loop that read_into_numpy_batch replaces"""
    for source, row_group, row_start, rows in tasks:
        column_names, target_column_indices = column_sets[source_column_set[source]]
        metadata = pj.read_metadata(
            index_paths[source], row_groups=[row_group], column_names=column_names
        )
        jj.read_into_numpy(
            source=sources[source],
            metadata=metadata,
            np_array=np_array[row_start : row_start + rows, :],
            row_group_indices=[0],
            column_names=list(zip(column_names, target_column_indices)),
            pre_buffer=True,
            use_threads=False,
        )


class TestReadIntoNumpyBatch(unittest.TestCase):

    def setUp(self):
        self.tmpdirname = self.enterContext(tempfile.TemporaryDirectory())
        self.index_buffers = []
        # A mapped index cannot be deleted on every file system, so unmap before the directory is removed
        self.addCleanup(
            lambda: [index_buffer.close() for index_buffer in self.index_buffers]
        )
        # io_uring is not available everywhere, so read with the default reader whatever the environment asks for
        self.backend = "default"
        os.environ.pop("JJ_READER_BACKEND", None)

    def run_batch(self, column_sets, n_columns, n_threads=4):
        """Write one parquet file per column set, then read every row group of every file, both ways"""
        sources, index_paths, source_column_set = [], [], []
        for i, (column_names, _) in enumerate(column_sets):
            path = os.path.join(self.tmpdirname, f"my_{i}.parquet")
            index_paths.append(write_parquet(path, column_names))
            sources.append(path)
            source_column_set.append(i)
            with open(index_paths[-1], "rb") as f:
                self.index_buffers.append(
                    mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
                )

        tasks = np.array(
            [
                (source, rg, rg * chunk_size, chunk_size)
                for rg in range(n_row_groups)
                for source in range(len(sources))
            ],
            dtype=np.int64,
        )

        expected = np.zeros((n_rows, n_columns), dtype="f", order="F")
        read_task_by_task(
            expected, sources, index_paths, column_sets, source_column_set, tasks
        )

        np_array = np.zeros((n_rows, n_columns), dtype="f", order="F")
        self.read_batch(
            np_array,
            sources,
            column_sets,
            source_column_set,
            tasks,
            n_threads=n_threads,
        )
        return np_array, expected, sources, source_column_set, tasks

    def read_batch(
        self, np_array, sources, column_sets, source_column_set, tasks, n_threads=4
    ):
        jj.read_into_numpy_batch(
            np_array=np_array,
            sources=sources,
            index_buffers=self.index_buffers,
            column_sets=column_sets,
            source_column_set=source_column_set,
            tasks=tasks,
            read_metadata_capi=pj._read_metadata_capi,
            n_threads=n_threads,
            pre_buffer=True,
            use_threads=False,
            backend=self.backend,
        )

    def test_single_column_set(self):
        column_sets = [([f"column_{i}" for i in range(4)], [0, 1, 2, 3])]
        np_array, expected, *_ = self.run_batch(column_sets, n_columns=4)
        self.assertTrue(np.array_equal(np_array, expected))
        self.assertTrue(np_array.any())

    def test_multiple_column_sets(self):
        column_sets = [
            ([f"column_{i}" for i in range(3)], [0, 1, 2]),
            ([f"other_{i}" for i in range(2)], [3, 4]),
        ]
        np_array, expected, *_ = self.run_batch(column_sets, n_columns=5)
        self.assertTrue(np.array_equal(np_array, expected))
        self.assertTrue(np_array.any())

    def test_non_contiguous_target_indices(self):
        column_sets = [
            ([f"column_{i}" for i in range(3)], [5, 0, 3]),
            ([f"other_{i}" for i in range(2)], [4, 1]),
        ]
        np_array, expected, *_ = self.run_batch(column_sets, n_columns=6)
        self.assertTrue(np.array_equal(np_array, expected))
        # Column 2 is in no column set, so it is left untouched
        self.assertFalse(np_array[:, 2].any())
        self.assertTrue(np_array[:, [0, 1, 3, 4, 5]].all())

    def test_single_thread(self):
        column_sets = [([f"column_{i}" for i in range(4)], [0, 1, 2, 3])]
        np_array, expected, *_ = self.run_batch(column_sets, n_columns=4, n_threads=1)
        self.assertTrue(np.array_equal(np_array, expected))

    def test_missing_source(self):
        column_sets = [([f"column_{i}" for i in range(4)], [0, 1, 2, 3])]
        _, _, sources, source_column_set, tasks = self.run_batch(
            column_sets, n_columns=4
        )
        os.remove(sources[0])
        np_array = np.zeros((n_rows, 4), dtype="f", order="F")
        with self.assertRaises(RuntimeError) as context:
            self.read_batch(
                np_array, sources, column_sets, source_column_set, tasks[1:2]
            )

        self.assertTrue(
            f"Failed to read row group 1 of '{sources[0]}'" in str(context.exception),
            context.exception,
        )

    def test_tasks_out_of_bounds(self):
        column_sets = [([f"column_{i}" for i in range(4)], [0, 1, 2, 3])]
        _, _, sources, source_column_set, tasks = self.run_batch(
            column_sets, n_columns=4
        )
        np_array = np.zeros((n_rows - 1, 4), dtype="f", order="F")
        with self.assertRaises(AssertionError):
            self.read_batch(np_array, sources, column_sets, source_column_set, tasks)

    def test_unknown_backend(self):
        column_sets = [([f"column_{i}" for i in range(4)], [0, 1, 2, 3])]
        _, _, sources, source_column_set, tasks = self.run_batch(
            column_sets, n_columns=4
        )
        np_array = np.zeros((n_rows, 4), dtype="f", order="F")
        self.backend = "not_a_backend"
        with self.assertRaises(ValueError):
            self.read_batch(np_array, sources, column_sets, source_column_set, tasks)

    def test_wrong_capsule(self):
        column_sets = [([f"column_{i}" for i in range(4)], [0, 1, 2, 3])]
        _, _, sources, source_column_set, tasks = self.run_batch(
            column_sets, n_columns=4
        )
        np_array = np.zeros((n_rows, 4), dtype="f", order="F")
        with self.assertRaises(ValueError):
            jj.read_into_numpy_batch(
                np_array=np_array,
                sources=sources,
                index_buffers=self.index_buffers,
                column_sets=column_sets,
                source_column_set=source_column_set,
                tasks=tasks,
                read_metadata_capi=object(),
                n_threads=4,
            )


if __name__ == "__main__":
    unittest.main()
