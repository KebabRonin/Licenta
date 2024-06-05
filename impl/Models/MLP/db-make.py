import polars as pl
# from my_utils import *
from tqdm import trange
import psycopg2, random,sys, time
sys.stdout.reconfigure(encoding='utf-8')
# a = pl.DataFrame({'a': [i for i in range(100)], 'b': [i for i in range(100)]})
# print(a.schema)
# a.write_database('test', connection='postgresql://postgres:admin@localhost:5432/Data', if_table_exists='append')
# pl.scan_parquet("Dataset/train/v1/train_0.parquet").slice(offset=170_000, length=1).collect()
# pl.scan_parquet("Dataset/train/v1/train_0.parquet").filter(pl.col('sample_id') == 'train_170000').collect()
# CustomSQLDataset(norm_method="none")[170_000]
# from torch.utils.data import RandomSampler
# exit(0)
# t0 = time.time()
# for i in trange(2_000):
# 	file = random.randint(0, 51)
# 	offset = random.randint(0, 100_000)
# 	pl.scan_parquet(f"Dataset/train/v1/train_{file}.parquet").slice(offset=offset).head(1).collect()
# print(time.time() - t0)
# exit(0)
# psycopg2.
# conn = psycopg2.connect(
# 		host="localhost",
# 		database="Data",
# 		user="postgres",
# 		password="admin"
# 	)
# cursor = conn.cursor()
# i = 5758851
# cursor.execute(f"DELETE FROM public.train WHERE ctid IN (select ctid from public.train where sample_id_int = {i} limit 1)")
# conn.commit()
# exit(0)
# def th_fn(start, end, id):
# 	conn = psycopg2.connect(
# 		host="localhost",
# 		database="Data",
# 		user="postgres",
# 		password="admin"
# 	)
# 	i = 0
# 	lg = 100
# 	for i in trange(start, end, miniters=500, position=id, maxinterval=200):
# 		b = pl.read_database(f"select sample_id_int from public.train where sample_id_int = {i}", connection=conn)
# 		lg = b.select(pl.len()).item()
# 		if lg != 1:
# 			print(i, lg)
# import threading
# ths = [threading.Thread(target=th_fn, args=(i*500_000, (i+1)*500_000, i)) for i in range(20)]
# for th in ths:
# 	th.start()
# exit(0)
# b = pl.read_database(f"select count(sample_id) from public.test", connection=conn)
# print(b)
# b = pl.read_database(f"select sample_id from public.test where sample_id = 'train_8599999'", connection=conn)
# print(b)
# exit(0)
# 8765945
# a = pl.scan_parquet(f"Dataset/train/v1/train_27.parquet").slice(offset=150000).collect()
# a.write_database('test', connection='postgresql://postgres:admin@localhost:5432/Data', if_table_exists='append')
from torch.utils.data import DataLoader, Dataset
def main():
	class CustomSQLDataset(Dataset):
		def __init__(self, norm_method = "+mean/std"):
			self.norm_method = norm_method

		def __len__(self):
			return 8_750_000

		def __getitem__(self, idx):
			# df = cursor.execute(f"select * from public.test where sample_id = 'train_{idx}'")
			df = pl.read_database(f"select * from public.test where sample_id = 'train_{idx}'", connection=conn)
			# image = df[in_vars].to_numpy()
			features = normalize_subset(df, in_vars, method=self.norm_method).to_numpy().squeeze()
			target = normalize_subset(df, out_vars, method=self.norm_method).to_numpy().squeeze()
			# cursor.execute(f"select * from public.test where sample_id = 'train_{idx}'")
			# df = np.array(cursor.fetchone())[1:]
			# print(df)
			# image = df[:556]
			# label = df[556:]
			return features, target
	DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
	sqldloader = DataLoader(CustomSQLDataset(), batch_size=1_000, shuffle=True, num_workers=0, pin_memory=True, pin_memory_device=DEVICE)
# 	import time
# 	print("Start")
# 	t1 = time.time()
# 	xs, ys = next(iter(sqldloader))
# 	plt.plot(xs[None, :10])
# 	plt.legend
# 	# plt.plot(ys)

# 	plt.show()
# 	print(xs.shape)
# 	print(time.time() - t1)

# if __name__ == "__main__":
# 	main()

# exit(0)
# b = pl.DataFrame({'a': [1]})
# i = 8_750_000
# while b.select(pl.len()).item() == 1:
# 	b = pl.read_database(f"select sample_id from public.test where sample_id = 'train_{i}'", connection=conn)
# 	print(i)
# 	i += 10000
# print(b, b.select(pl.len()).item())
# print(i)
# a = [pl.scan_parquet(f"Dataset/train/v1/train_{i}.parquet").select(pl.len()).collect().item() for i in range(51)]
# s = 0
# for i, v in enumerate(a):
# 	s += v
# 	print(i, s)
# exit(0)
# # b.write_database('test', connection='postgresql://postgres:admin@localhost:5432/Data', if_table_exists='append')

# # import matplotlib.pyplot as plt
# # a = pl.scan_parquet(f"Dataset/train/v1/train_15.parquet").slice(offset=150_000, length=1).collect()
# 	# plt.plot(a['ptend_q0001_50'])
# 	# plt.show()
# exit(0)
# file_rows = 100
# left_to_do = file_rows
# file_nr = 0
# schema = pl.scan_parquet("Dataset/train/v1/train_0.parquet").drop('sample_id').schema
# remainder = pl.DataFrame(schema=schema)
# exit(0)
# for i in trange(51, desc='files'):
# 	ff = pl.scan_parquet(f"Dataset/train/v1/train_{i}.parquet").drop('sample_id')
# 	l = ff.select(pl.len()).collect().item()
# 	left_to_do = file_rows
# 	last_offset = 0
# 	while last_offset < l:
# 		this_time = ff.slice(last_offset, file_rows)
# 		last_offset += this_time.select(pl.len()).collect().item()

# 		remainder = pl.concat([remainder, this_time.collect()]).write_parquet(f"Dataset/train/v1/train_{file_nr}.parquet")


# last written: 3 + 9
# last row: 8765944
# a = pl.scan_parquet(f"Dataset/train/v1/train_43.parquet").slice(offset=8765945-8615945).collect()
# a.write_database('test', connection='postgresql://postgres:admin@localhost:5432/Data', if_table_exists='append')
# exit(0)
# batch_size = 50_000

# for i in trange(44, 51, desc='train files'):
# 	df = pl.scan_parquet(f"Dataset/train/v1/train_{i}.parquet")
# 	l = df.select(pl.len()).collect().item()
# 	for batch_start in trange(0, l, batch_size, desc='batches'):
# 		batch = df.slice(batch_start, batch_size).collect()
# 		le = batch.select(pl.len()).item()
# 		if le < batch_size:
# 			print(le)

# 		batch.write_database('test', connection='postgresql://postgres:admin@localhost:5432/Data', if_table_exists='append')