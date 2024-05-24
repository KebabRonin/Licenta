Dataset taken from [Giba](https://www.kaggle.com/datasets/titericz/leap-dataset-giba)


Data len: 10_091_520

> Now you can load LEAP dataset fast and efficiently.
> This is the full competition train dataset split in 17 parquet files. Test set is a single file.
> Features in *float16* format. Target in *float32* to avoid losing precision due quantization.
> To load just use pandas.read_parquet(filename) or cudf.read_parquet(filename) methods.

Columns with 1 unique value dropped: \[
	pbuf_CH4_27-59,
	pbuf_N2O_27-59
]


TODO:
Add issue to torch: \[optim] add missing 'maximize' parameter to LBFGS, NAdam and RAdam optimizers
Investigate polars speed for LazyFrame .slice(offset, batch_size) vs .shift(offset).limit(batch_size) vs lf.fetch() vs with lf.profile()