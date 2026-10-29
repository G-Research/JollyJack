import ctypes
import itertools
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


def write_parquet(path, column_names, n_row_groups=n_row_groups):
    data = np.random.uniform(
        -100, 100, size=(n_row_groups * chunk_size, len(column_names))
    ).astype(np.float32)
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
    np_array,
    sources,
    index_paths,
    column_sets,
    source_column_set,
    tasks,
    pre_buffer,
    use_threads,
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
            pre_buffer=pre_buffer,
            use_threads=use_threads,
        )


class TestReadIntoNumpyBatch(unittest.TestCase):

    def setUp(self):
        self.tmpdirname = self.enterContext(tempfile.TemporaryDirectory())
        self.index_buffers = []
        # A mapped index cannot be deleted on every file system, so unmap before the directory is removed
        self.addCleanup(
            lambda: [index_buffer.close() for index_buffer in self.index_buffers]
        )

    def write_sources(self, column_sets, source_column_set=None, n_row_groups=n_row_groups):
        """Write one parquet file per source, with the columns of its column set (by default one source per column
        set), and a task for every row group of every file, placed at the rows of the row group"""
        if source_column_set is None:
            source_column_set = list(range(len(column_sets)))
        sources, index_paths = [], []
        for i, column_set in enumerate(source_column_set):
            path = os.path.join(self.tmpdirname, f"my_{i}.parquet")
            index_paths.append(
                write_parquet(path, column_sets[column_set][0], n_row_groups)
            )
            sources.append(path)
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
        return sources, index_paths, source_column_set, tasks

    def read_both(
        self,
        sources,
        index_paths,
        column_sets,
        source_column_set,
        tasks,
        n_columns,
        n_threads=4,
        pre_buffer=True,
        use_threads=False,
    ):
        """Read the tasks task by task and as a batch, into arrays just tall enough for them"""
        rows = int((tasks[:, 2] + tasks[:, 3]).max())
        expected = np.zeros((rows, n_columns), dtype="f", order="F")
        read_task_by_task(
            expected,
            sources,
            index_paths,
            column_sets,
            source_column_set,
            tasks,
            pre_buffer,
            use_threads,
        )

        np_array = np.zeros((rows, n_columns), dtype="f", order="F")
        self.read_batch(
            np_array,
            sources,
            column_sets,
            source_column_set,
            tasks,
            n_threads=n_threads,
            pre_buffer=pre_buffer,
            use_threads=use_threads,
        )
        return np_array, expected

    def run_batch(
        self, column_sets, n_columns, n_threads=4, pre_buffer=True, use_threads=False
    ):
        """Write the sources, then read every row group of every file, both ways"""
        sources, index_paths, source_column_set, tasks = self.write_sources(
            column_sets
        )
        np_array, expected = self.read_both(
            sources,
            index_paths,
            column_sets,
            source_column_set,
            tasks,
            n_columns,
            n_threads=n_threads,
            pre_buffer=pre_buffer,
            use_threads=use_threads,
        )
        return np_array, expected, sources, source_column_set, tasks

    def read_batch(
        self,
        np_array,
        sources,
        column_sets,
        source_column_set,
        tasks,
        n_threads=4,
        pre_buffer=True,
        use_threads=False,
    ):
        jj.read_into_numpy_batch(
            np_array=np_array,
            sources=sources,
            index_buffers=self.index_buffers,
            column_sets=column_sets,
            source_column_set=source_column_set,
            tasks=tasks,
            palletjack_capi=pj.palletjack_capi,
            n_threads=n_threads,
            pre_buffer=pre_buffer,
            use_threads=use_threads,
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

    def test_read_options(self):
        column_sets = [([f"column_{i}" for i in range(4)], [0, 1, 2, 3])]
        np_array, expected, sources, source_column_set, tasks = self.run_batch(
            column_sets, n_columns=4
        )
        self.assertTrue(np.array_equal(np_array, expected))
        for pre_buffer, use_threads in itertools.product([False, True], [False, True]):
            with self.subTest(pre_buffer=pre_buffer, use_threads=use_threads):
                np_array = np.zeros((n_rows, 4), dtype="f", order="F")
                self.read_batch(
                    np_array,
                    sources,
                    column_sets,
                    source_column_set,
                    tasks,
                    pre_buffer=pre_buffer,
                    use_threads=use_threads,
                )
                self.assertTrue(np.array_equal(np_array, expected))

    def test_rows_placed_by_task(self):
        column_sets = [([f"column_{i}" for i in range(4)], [0, 1, 2, 3])]
        sources, index_paths, source_column_set, _ = self.write_sources(column_sets)
        # Row group 1 goes to the first rows and row group 0 after it, so the rows follow row_start, not the row group
        tasks = np.array(
            [(0, 1, 0, chunk_size), (0, 0, chunk_size, chunk_size)], dtype=np.int64
        )
        np_array, expected = self.read_both(
            sources, index_paths, column_sets, source_column_set, tasks, n_columns=4
        )
        self.assertTrue(np.array_equal(np_array, expected))
        table = pq.read_table(sources[0])
        self.assertTrue(
            np.array_equal(np_array[:chunk_size, 0], table["column_0"][chunk_size:])
        )

    def test_shared_column_set(self):
        column_sets = [([f"column_{i}" for i in range(4)], [0, 1, 2, 3])]
        sources, index_paths, source_column_set, _ = self.write_sources(
            column_sets, source_column_set=[0, 0, 0]
        )
        tasks = np.array(
            [
                (source, rg, (source * n_row_groups + rg) * chunk_size, chunk_size)
                for source in range(len(sources))
                for rg in range(n_row_groups)
            ],
            dtype=np.int64,
        )
        np_array, expected = self.read_both(
            sources, index_paths, column_sets, source_column_set, tasks, n_columns=4
        )
        self.assertTrue(np.array_equal(np_array, expected))
        self.assertTrue(np_array.all())

    def test_many_tasks(self):
        column_sets = [
            ([f"column_{i}" for i in range(4)], [0, 1, 2, 3]),
            ([f"other_{i}" for i in range(2)], [4, 5]),
        ]
        many_row_groups = 25
        sources, index_paths, source_column_set, _ = self.write_sources(
            column_sets, source_column_set=[0, 1, 0, 1], n_row_groups=many_row_groups
        )
        # Every task gets its own rows, in a shuffled order, so the threads pick them up out of order
        tasks = [
            (source, rg, (source * many_row_groups + rg) * chunk_size, chunk_size)
            for source in range(len(sources))
            for rg in range(many_row_groups)
        ]
        np.random.default_rng(0).shuffle(tasks)
        tasks = np.array(tasks, dtype=np.int64)
        for n_threads in [3, 8, 200]:
            with self.subTest(n_threads=n_threads):
                np_array, expected = self.read_both(
                    sources,
                    index_paths,
                    column_sets,
                    source_column_set,
                    tasks,
                    n_columns=6,
                    n_threads=n_threads,
                )
                self.assertTrue(np.array_equal(np_array, expected))

    def test_failure_among_many_tasks(self):
        column_sets = [([f"column_{i}" for i in range(4)], [0, 1, 2, 3])]
        many_row_groups = 25
        sources, _, source_column_set, _ = self.write_sources(
            column_sets, source_column_set=[0, 0], n_row_groups=many_row_groups
        )
        tasks = np.array(
            [
                (source, rg, (rg * 2 + source) * chunk_size, chunk_size)
                for rg in range(many_row_groups)
                for source in range(len(sources))
            ],
            dtype=np.int64,
        )
        os.remove(sources[1])
        np_array = np.zeros((len(tasks) * chunk_size, 4), dtype="f", order="F")
        with self.assertRaises(RuntimeError) as context:
            self.read_batch(
                np_array, sources, column_sets, source_column_set, tasks, n_threads=8
            )

        self.assertTrue("Failed to read row group" in str(context.exception), context.exception)
        self.assertTrue(f"of '{sources[1]}'" in str(context.exception), context.exception)

    def test_task_rows_differ_from_row_group(self):
        column_sets = [([f"column_{i}" for i in range(4)], [0, 1, 2, 3])]
        sources, _, source_column_set, _ = self.write_sources(column_sets)
        for rows, error in [
            (chunk_size - 1, "Buffer overrun error"),
            (chunk_size + 1, f"Expected to read {chunk_size + 1} rows"),
        ]:
            with self.subTest(rows=rows):
                tasks = np.array([(0, 0, 0, rows)], dtype=np.int64)
                np_array = np.zeros((n_rows, 4), dtype="f", order="F")
                with self.assertRaises(RuntimeError) as context:
                    self.read_batch(
                        np_array, sources, column_sets, source_column_set, tasks
                    )

                self.assertTrue(error in str(context.exception), context.exception)

    def test_row_group_out_of_range(self):
        column_sets = [([f"column_{i}" for i in range(4)], [0, 1, 2, 3])]
        sources, _, source_column_set, _ = self.write_sources(column_sets)
        tasks = np.array([(0, n_row_groups, 0, chunk_size)], dtype=np.int64)
        np_array = np.zeros((n_rows, 4), dtype="f", order="F")
        with self.assertRaises(RuntimeError) as context:
            self.read_batch(np_array, sources, column_sets, source_column_set, tasks)

        self.assertTrue(
            f"Failed to read row group {n_row_groups} of '{sources[0]}'"
            in str(context.exception),
            context.exception,
        )

    def test_unknown_column(self):
        column_sets = [([f"column_{i}" for i in range(4)], [0, 1, 2, 3])]
        sources, _, source_column_set, tasks = self.write_sources(column_sets)
        np_array = np.zeros((n_rows, 4), dtype="f", order="F")
        with self.assertRaises(RuntimeError) as context:
            self.read_batch(
                np_array,
                sources,
                [(["column_0", "not_a_column"], [0, 1])],
                source_column_set,
                tasks,
            )

        self.assertTrue(
            "Failed to read row group" in str(context.exception), context.exception
        )
        self.assertTrue("not_a_column" in str(context.exception), context.exception)

    def test_no_tasks(self):
        column_sets = [([f"column_{i}" for i in range(4)], [0, 1, 2, 3])]
        sources, _, source_column_set, _ = self.write_sources(column_sets)
        tasks = np.empty((0, 4), dtype=np.int64)
        np_array = np.zeros((n_rows, 4), dtype="f", order="F")
        self.read_batch(np_array, sources, column_sets, source_column_set, tasks)
        self.assertFalse(np_array.any())

        # The capsule is checked even when there is nothing to read
        with self.assertRaises(ValueError):
            jj.read_into_numpy_batch(
                np_array=np_array,
                sources=sources,
                index_buffers=self.index_buffers,
                column_sets=column_sets,
                source_column_set=source_column_set,
                tasks=tasks,
                palletjack_capi=object(),
                n_threads=4,
            )

    def test_target_column_out_of_range(self):
        column_sets = [([f"column_{i}" for i in range(4)], [0, 1, 2, 3])]
        sources, _, source_column_set, tasks = self.write_sources(column_sets)
        for target in [4, -1]:
            with self.subTest(target=target):
                np_array = np.zeros((n_rows, 4), dtype="f", order="F")
                with self.assertRaises(RuntimeError) as context:
                    self.read_batch(
                        np_array,
                        sources,
                        [(column_sets[0][0], [0, 1, 2, target])],
                        source_column_set,
                        tasks,
                    )

                self.assertTrue(
                    "Buffer overrun error" in str(context.exception), context.exception
                )

    def test_missing_source(self):
        column_sets = [([f"column_{i}" for i in range(4)], [0, 1, 2, 3])]
        sources, _, source_column_set, tasks = self.write_sources(column_sets)
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
        sources, _, source_column_set, tasks = self.write_sources(column_sets)
        np_array = np.zeros((n_rows - 1, 4), dtype="f", order="F")
        with self.assertRaises(AssertionError):
            self.read_batch(np_array, sources, column_sets, source_column_set, tasks)

    def test_unknown_backend(self):
        column_sets = [([f"column_{i}" for i in range(4)], [0, 1, 2, 3])]
        sources, _, source_column_set, tasks = self.write_sources(column_sets)
        np_array = np.zeros((n_rows, 4), dtype="f", order="F")
        jj_backend = os.environ.get("JJ_READER_BACKEND")
        os.environ["JJ_READER_BACKEND"] = "not_a_backend"
        try:
            with self.assertRaises(ValueError) as context:
                self.read_batch(
                    np_array, sources, column_sets, source_column_set, tasks
                )
        finally:
            if jj_backend is None:
                os.environ.pop("JJ_READER_BACKEND")
            else:
                os.environ["JJ_READER_BACKEND"] = jj_backend

        self.assertTrue(
            "Unsupported JJ_READER_BACKEND=not_a_backend" in str(context.exception),
            context.exception,
        )

    def test_wrong_capsule(self):
        column_sets = [([f"column_{i}" for i in range(4)], [0, 1, 2, 3])]
        sources, _, source_column_set, tasks = self.write_sources(column_sets)
        np_array = np.zeros((n_rows, 4), dtype="f", order="F")
        with self.assertRaises(ValueError):
            jj.read_into_numpy_batch(
                np_array=np_array,
                sources=sources,
                index_buffers=self.index_buffers,
                column_sets=column_sets,
                source_column_set=source_column_set,
                tasks=tasks,
                palletjack_capi=object(),
                n_threads=4,
            )

    def test_old_capi_version(self):
        column_sets = [([f"column_{i}" for i in range(4)], [0, 1, 2, 3])]
        sources, _, source_column_set, tasks = self.write_sources(column_sets)
        np_array = np.zeros((n_rows, 4), dtype="f", order="F")
        # A local prototype, setting argtypes on ctypes.pythonapi.PyCapsule_New would change it for the whole process
        capsule_new = ctypes.PYFUNCTYPE(
            ctypes.py_object, ctypes.c_void_p, ctypes.c_char_p, ctypes.c_void_p
        )(("PyCapsule_New", ctypes.pythonapi))
        # The capsule keeps pointers to the name and the table, both must outlive it
        name = ctypes.c_char_p(b"palletjack.palletjack_capi")
        table = (ctypes.c_void_p * 4)()  # version 0, no members
        capsule = capsule_new(ctypes.addressof(table), name, None)
        with self.assertRaises(ValueError):
            jj.read_into_numpy_batch(
                np_array=np_array,
                sources=sources,
                index_buffers=self.index_buffers,
                column_sets=column_sets,
                source_column_set=source_column_set,
                tasks=tasks,
                palletjack_capi=capsule,
                n_threads=4,
            )


if __name__ == "__main__":
    unittest.main()
