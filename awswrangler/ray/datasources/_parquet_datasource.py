"""Ray Parquet Datasource Module (PRIVATE)."""

import ast
import logging
from typing import TYPE_CHECKING, Any, Callable, Dict, Iterator, List, Optional, Union

from awswrangler.s3._read import _extract_partitions_from_path

if TYPE_CHECKING:
    import pyarrow

from ray.data.block import Block, BlockAccessor
from ray.data.context import DatasetContext
from ray.data.datasource.datasource import ReadTask
from ray.data.datasource.file_based_datasource import (
    FileBasedDatasource,
    _resolve_kwargs,
    _resolve_paths_and_filesystem,
)
from ray.data.datasource.parquet_datasource import (
    DefaultParquetMetadataProvider,
    ParquetMetadataProvider,
    _deregister_parquet_file_fragment_serialization,
    _register_parquet_file_fragment_serialization,
)
from ray.data.impl.output_buffer import BlockOutputBuffer
from ray.data.impl.util import _check_pyarrow_version

logger = logging.getLogger(__name__)

PIECES_PER_META_FETCH = 6
PARALLELIZE_META_FETCH_THRESHOLD = 24

# The number of rows to read per batch. This is sized to generate 10MiB batches
# for rows about 1KiB in size.
PARQUET_READER_ROW_BATCH_SIZE = 100000


class ParquetDatasource(FileBasedDatasource):
    """Parquet datasource, for reading and writing Parquet files.

    Examples:
        >>> source = ParquetDatasource()
        >>> ray.data.read_datasource(source, paths="/path/to/dir").take()
        ... [{"a": 1, "b": "foo"}, ...]
    """

    def prepare_read(
        self,
        parallelism: int,
        paths: Union[str, List[str]],
        filesystem: Optional["pyarrow.fs.FileSystem"] = None,
        columns: Optional[List[str]] = None,
        schema: Optional[Union[type, "pyarrow.lib.Schema"]] = None,
        meta_provider: ParquetMetadataProvider = DefaultParquetMetadataProvider(),
        _block_udf: Optional[Callable[[Block], Block]] = None,
        path_root: Optional[str] = None,
        **reader_args,
    ) -> List[ReadTask]:
        """Creates and returns read tasks for a Parquet file-based datasource."""
        # NOTE: We override the base class FileBasedDatasource.prepare_read
        # method in order to leverage pyarrow's ParquetDataset abstraction,
        # which simplifies partitioning logic. We still use
        # FileBasedDatasource's write side (do_write), however.
        _check_pyarrow_version()
        import numpy as np
        import pyarrow as pa
        import pyarrow.parquet as pq

        from ray import cloudpickle

        paths, filesystem = _resolve_paths_and_filesystem(paths, filesystem)
        if len(paths) == 1:
            paths = paths[0]

        dataset_kwargs = reader_args.pop("dataset_kwargs", {})
        pq_ds = pq.ParquetDataset(
            paths,
            **dataset_kwargs,
            filesystem=filesystem,
            use_legacy_dataset=False,
            partitioning=None,
        )
        if schema is None:
            schema = (
                pq.ParquetDataset(
                    path_root,
                    **dataset_kwargs,
                    filesystem=filesystem,
                    use_legacy_dataset=False,
                    partitioning="hive",
                ).schema
                if path_root
                else pq_ds.schema
            )
        if columns:
            schema = pa.schema([schema.field(column) for column in columns], schema.metadata)

        def try_eval(val: Any) -> Any:
            try:
                val = ast.literal_eval(val)
            except ValueError:
                pass
            return val

        def read_pieces(serialized_pieces: str) -> Iterator[pa.Table]:
            # Implicitly trigger S3 subsystem initialization by importing
            # pyarrow.fs.
            import pyarrow.fs  # noqa: F401

            # Deserialize after loading the filesystem class.
            try:
                _register_parquet_file_fragment_serialization()
                pieces: List["pyarrow._dataset.ParquetFileFragment"] = cloudpickle.loads(serialized_pieces)
            finally:
                _deregister_parquet_file_fragment_serialization()

            # Ensure that we're reading at least one dataset fragment.
            assert len(pieces) > 0

            from pyarrow.dataset import _get_partition_keys

            ctx = DatasetContext.get_current()
            output_buffer = BlockOutputBuffer(block_udf=_block_udf, target_max_block_size=ctx.target_max_block_size)

            logger.debug(f"Reading {len(pieces)} parquet pieces")
            use_threads = reader_args.pop("use_threads", False)
            for piece in pieces:
                if path_root:
                    part = _extract_partitions_from_path(path_root, f"s3://{piece.path}")
                else:
                    part = _get_partition_keys(piece.partition_expression)
                batches = piece.to_batches(
                    use_threads=use_threads,
                    columns=columns,
                    schema=schema,
                    batch_size=PARQUET_READER_ROW_BATCH_SIZE,
                    **reader_args,
                )
                for batch in batches:
                    table = pyarrow.Table.from_batches([batch], schema=schema)
                    if part:
                        for col, value in part.items():
                            table = table.set_column(
                                table.schema.get_field_index(col),
                                col,
                                pa.array([try_eval(value)] * len(table)),
                            )
                    # If the table is empty, drop it.
                    if table.num_rows > 0:
                        output_buffer.add_block(table)
                        if output_buffer.has_next():
                            yield output_buffer.next()
            output_buffer.finalize()
            if output_buffer.has_next():
                yield output_buffer.next()

        if _block_udf is not None:
            # Try to infer dataset schema by passing dummy table through UDF.
            dummy_table = schema.empty_table()
            try:
                inferred_schema = _block_udf(dummy_table).schema
                inferred_schema = inferred_schema.with_metadata(schema.metadata)
            except Exception:
                logger.debug(
                    "Failed to infer schema of dataset by passing dummy table "
                    "through UDF due to the following exception:",
                    exc_info=True,
                )
                inferred_schema = schema
        else:
            inferred_schema = schema
        read_tasks = []
        metadata = meta_provider.prefetch_file_metadata(pq_ds.pieces) or []
        try:
            _register_parquet_file_fragment_serialization()
            for pieces, metadata in zip(
                np.array_split(pq_ds.pieces, parallelism),
                np.array_split(metadata, parallelism),
            ):
                if len(pieces) <= 0:
                    continue
                serialized_pieces = cloudpickle.dumps(pieces)
                input_files = [p.path for p in pieces]
                meta = meta_provider(
                    input_files,
                    inferred_schema,
                    pieces=pieces,
                    prefetched_metadata=metadata,
                )
                read_tasks.append(ReadTask(lambda p=serialized_pieces: read_pieces(p), meta))
        finally:
            _deregister_parquet_file_fragment_serialization()

        return read_tasks

    def _write_block(
        self,
        f: "pyarrow.NativeFile",
        block: BlockAccessor,
        writer_args_fn: Callable[[], Dict[str, Any]] = lambda: {},
        **writer_args,
    ):
        import pyarrow.parquet as pq

        writer_args = _resolve_kwargs(writer_args_fn, **writer_args)
        pq.write_table(block.to_arrow(), f, **writer_args)

    def _file_format(self) -> str:
        return "parquet"
