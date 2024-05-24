import sys
sys.stdout.reconfigure(encoding='utf-8')

# import os, torch, cloudpickle
# base_path = r"G:/Licenta/impl/Models/MLP/Saved/"
# fs = tuple(os.walk(base_path))
# print(fs)
# fs = sorted(fs[0][2])
# for f in fs:
# 	try:
# 		m = torch.load(base_path + f)
# 		print(f)
# 		# print(m)
# 		# torch.save(m, base_path + 'x' + f, pickle_module=cloudpickle)
# 	except Exception as e:
# 		# print(e)
# 		...

import polars as pl, time
def time_f(f):
	t0 = time.time()
	f()
	return time.time() - t0

f1 = lambda: print(pl.scan_parquet(r"Dataset/train/v1/train_1.parquet").slice(100_000,1).collect())
f2 = lambda: print(pl.scan_parquet(r"Dataset/train/mega/train.parquet").limit(100_001).last(1).collect())
f3 = lambda: print(pl.scan_parquet(r"Dataset/train/v1/train_1.parquet").shift(-100_000).limit(1).collect())
import pyarrow
with pyarrow.input_stream("Dataset/train/mega/train.parquet") as stream:
	print(stream.read(10))
# print("slice:", time_f(f1))
# print("shift:", time_f(f2))
# print("shift:", time_f(f3))
# print(pl.scan_parquet(r"Dataset/train/mega/train.parquet").slice(100_000,1).explain())
# print(pl.scan_parquet(r"Dataset/train/v1/train_1.parquet").schema)
# cc.sink_parquet("Dataset/train/mega/train.parquet")
# l = 10_091_520
# quit(0)
# for i in range(19):
# 	print(f"{i} > {i*lines_per_file}, ({(i+1)*lines_per_file-1})")
# 	cc.slice(i*lines_per_file, lines_per_file).sink_parquet(f"Dataset/train/train_{i}.parquet")
# else:
# 	print(f"{i} > {i*lines_per_file}, ({l}) {l - i*lines_per_file}")
# 	cc.slice(i*lines_per_file).sink_parquet(f"Dataset/train/train_19.parquet")