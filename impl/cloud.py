import polars as pl

pl.read_csv(r'G:\download\test.csv').write_parquet(r'test.parquet')