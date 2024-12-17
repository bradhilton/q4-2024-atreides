import polars as pl

df = pl.read_parquet(
    "hf://datasets/allenai/ZebraLogicBench-private/grid_mode/test-00000-of-00001.parquet"
)

df = pl.read_parquet(
    "hf://datasets/allenai/ZebraLogicBench-private/mc_mode/test-00000-of-00001.parquet"
)
