import asyncio
import sys
sys.stdout.reconfigure(encoding='utf-8')
x=0
async def f1():
	global x
	while x == 0:
		await asyncio.sleep(2)
		print(x)

async def f2():
	global x
	while True:
		a = input()
		x+=1

async def main():
	asyncio.create_task(f1())
	asyncio.create_task(f2())
# import time
# # asyncio.run(main())
# print('a', flush=True)
# time.sleep(5)
# print('b')

import polars as pl

lines_per_file = 500_000

# cc = pl.scan_parquet(r"Dataset/train/v1/train_1.parquet")
# print("Open")
# # print(cc.schema)
# print(cc.select(pl.len()).collect())

import os, torch, cloudpickle
base_path = "Saved/"
fs = tuple(os.walk(base_path))
print(fs)
fs = sorted(fs[0][2])
for f in fs:
	print(f)
	try:
		m = torch.load(base_path + f)
		# print(m)
		# torch.save(m, base_path + 'x' + f, pickle_module=cloudpickle)
	except Exception as e:
		print(e)

# cc.sink_parquet("Dataset/train/mega/train.parquet")
# l = 10_091_520
# quit(0)
# for i in range(19):
# 	print(f"{i} > {i*lines_per_file}, ({(i+1)*lines_per_file-1})")
# 	cc.slice(i*lines_per_file, lines_per_file).sink_parquet(f"Dataset/train/train_{i}.parquet")
# else:
# 	print(f"{i} > {i*lines_per_file}, ({l}) {l - i*lines_per_file}")
# 	cc.slice(i*lines_per_file).sink_parquet(f"Dataset/train/train_19.parquet")