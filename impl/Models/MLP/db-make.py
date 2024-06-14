import polars as pl, torch
from my_utils import *
from tqdm import trange, tqdm
import matplotlib.pyplot as plt
import psycopg2, random, sys, time
from mpl_toolkits.basemap import Basemap
import xarray as xr
import cloudpickle as cp

TOTAL_SAMPLES = 10_091_520
N_TRAIN_SAMPLES = 50_000 / TOTAL_SAMPLES
N_VALID_SAMPLES = 10_000 / TOTAL_SAMPLES

from model_def import MLP_60_split
from torchmetrics.regression import R2Score
import cloudpickle
def main():
	with torch.no_grad():
		r2loss = R2Score(num_outputs=368).to(DEVICE)
		model = MLP_60_split().to(DEVICE)
		model.eval()
		d = dict()
		for n in tqdm(['+mean', '+mean/std', 'minmax10', 'minmax100', 'mean norm']):
			torch.manual_seed(42)
			dset = CustomSQLDataset(norm_method=n)
			_, valid_data, _ = tdata.random_split(dset, [N_TRAIN_SAMPLES, N_VALID_SAMPLES, 1 - N_TRAIN_SAMPLES - N_VALID_SAMPLES], generator=torch.Generator().manual_seed(50))
			valid_loader = DataLoader(
				valid_data,
				batch_sampler=tdata.BatchSampler(tdata.SequentialSampler(valid_data), batch_size=50_000, drop_last=False),
				collate_fn=identity,
				num_workers=1,
				persistent_workers=False,
				drop_last=False
			)

			for features, target in valid_loader:
				features, target = features.to(DEVICE), target.to(DEVICE)
				preds = model(features)
				init_loss = r2loss(preds, target).item()
				print(n, init_loss)
				d[n] = init_loss
	cloudpickle.dump(d, open('init_losses_norm_experiment_resnet5_1024.pkl', 'wb'))
if __name__ == '__main__':
	main()
# exit(0)
# # from model_def import MLP_60_split
# # model = MLP_60_split()
# # print(next(iter(model.parameters()))[:10])
# # torch.manual_seed(42)
# # model = MLP_60_split()
# # print(next(iter(model.parameters()))[:10])
# # torch.manual_seed(42)
# # model = MLP_60_split()
# # print(next(iter(model.parameters()))[:10])
# import glob
# for f in (glob.glob(r"*5_r2loss2.pkl")):
# 		try:
# 			cc = cp.load(open(f, 'rb'))
# 			# plt.figure()
# 			plt.plot(cc, label=f)
# 			# plt.title(f)
# 		except FileNotFoundError:
# 			print("ooopse", f)
# plt.legend()
# plt.axis((None, None, -1.5, 1.5))
# plt.show()
# exit(0)
# # file = r'C:\Users\KebabWarrior\Desktop\Facultate\ClimSim\grid_info\ClimSim_low-res_grid-info.nc'
# # grid = xr.open_dataset(file,engine='netcdf4')
# # x = grid.lon
# # y = grid.lat
# # m = Basemap(projection='robin',lon_0=165,resolution='c')
# # data = np.random.rand(len(x))
# # m.drawcoastlines(linewidth=0.5)
# # x,y = m(grid.lon,grid.lat)
# # plot = plt.tricontourf(x,y,data,cmap='viridis',levels=14,vmin=0.,vmax=1.)
# # m.colorbar(plot)
# # plt.show()
# # exit(0)
# # query = 'select ' + ','.join(f'min("{i}") as "{i}"' for i in (in_vars + out_vars)) + ' from public.train;'
# # a = pl.read_database(query, connection=conns.getconn())
# # for c in (in_vars + out_vars):
# # 	data_insights[c]['min'] = a[c].item()
# # json.dump(data_insights, open("tyrdjsfd.json", 'w'))
# # exit(0)
# sys.stdout.reconfigure(encoding='utf-8')
# a = pl.read_parquet("Dataset/train/v1/train_0.parquet", n_rows=5_000).drop('sample_id')
# c = 'ptend_q0002_25'
# plt.plot(a[c]) #.shift(-384))
# plt.plot(a['state_q0002_25']) #.shift(-384))
# plt.show()
# i, o = preprocess_standardisation(a.to_numpy())
# print("read", o.shape, a.select(pl.len()))
# valls = [i for i in range(5_000)]
# x = out_vars.index(c)
# plt.plot(o[:, x])
# plt.plot(i[:, in_vars.index('state_q0002_25')])
# plt.title(c)
# # plt.savefig(f"{c}.png")
# plt.show()
# # print(in_vars[360])
# # plt.plot(in_std_dev)
# # plt.plot(out_std_dev)
# # plt.show()
# exit(0)
# # a = pl.DataFrame({'a': [i for i in range(100)], 'b': [i for i in range(100)]})
# # print(a.schema)
# # a.write_database('test', connection='postgresql://postgres:admin@localhost:5432/Data', if_table_exists='append')
# # pl.scan_parquet("Dataset/train/v1/train_0.parquet").slice(offset=170_000, length=1).collect()
# # pl.scan_parquet("Dataset/train/v1/train_0.parquet").filter(pl.col('sample_id') == 'train_170000').collect()
# # CustomSQLDataset(norm_method="none")[170_000]
# # from torch.utils.data import RandomSampler
# # exit(0)
# # t0 = time.time()
# # for i in trange(2_000):
# # 	file = random.randint(0, 51)
# # 	offset = random.randint(0, 100_000)
# # 	pl.scan_parquet(f"Dataset/train/v1/train_{file}.parquet").slice(offset=offset).head(1).collect()
# # print(time.time() - t0)
# # exit(0)
# # psycopg2.
# # conn = psycopg2.connect(
# # 		host="localhost",
# # 		database="Data",
# # 		user="postgres",
# # 		password="admin"
# # 	)
# # cursor = conn.cursor()
# # i = 5758851
# # cursor.execute(f"DELETE FROM public.train WHERE ctid IN (select ctid from public.train where sample_id_int = {i} limit 1)")
# # conn.commit()
# # exit(0)
# # def th_fn(start, end, id):
# # 	conn = psycopg2.connect(
# # 		host="localhost",
# # 		database="Data",
# # 		user="postgres",
# # 		password="admin"
# # 	)
# # 	i = 0
# # 	lg = 100
# # 	for i in trange(start, end, miniters=500, position=id, maxinterval=200):
# # 		b = pl.read_database(f"select sample_id_int from public.train where sample_id_int = {i}", connection=conn)
# # 		lg = b.select(pl.len()).item()
# # 		if lg != 1:
# # 			print(i, lg)
# # import threading
# # ths = [threading.Thread(target=th_fn, args=(i*500_000, (i+1)*500_000, i)) for i in range(20)]
# # for th in ths:
# # 	th.start()
# # exit(0)
# # b = pl.read_database(f"select count(sample_id) from public.test", connection=conn)
# # print(b)
# # b = pl.read_database(f"select sample_id from public.test where sample_id = 'train_8599999'", connection=conn)
# # print(b)
# # exit(0)
# # 8765945
# # a = pl.scan_parquet(f"Dataset/train/v1/train_27.parquet").slice(offset=150000).collect()
# # a.write_database('test', connection='postgresql://postgres:admin@localhost:5432/Data', if_table_exists='append')
# from torch.utils.data import DataLoader, Dataset
# def main():
# 	DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
# 	sqldloader = DataLoader(CustomSQLDataset(), batch_size=1_000, shuffle=True, num_workers=0, pin_memory=True, pin_memory_device=DEVICE)
# # 	import time
# # 	print("Start")
# # 	t1 = time.time()
# # 	xs, ys = next(iter(sqldloader))
# # 	plt.plot(xs[None, :10])
# # 	plt.legend
# # 	# plt.plot(ys)

# # 	plt.show()
# # 	print(xs.shape)
# # 	print(time.time() - t1)
# import torch.utils.data as tdata
# from my_utils import *
# TOTAL_SAMPLES = 10_091_520
# N_TRAIN_SAMPLES = 150_000 / TOTAL_SAMPLES
# N_VALID_SAMPLES = 10_000 / TOTAL_SAMPLES
# dset = CustomSQLDataset(norm_method="none")
# train_data, _, _ = tdata.random_split(dset, [N_TRAIN_SAMPLES, N_VALID_SAMPLES, 1 - N_TRAIN_SAMPLES - N_VALID_SAMPLES], generator=torch.Generator().manual_seed(50))
# del _
# # print(sorted(train_data.indices))

# conn = psycopg2.connect(
# 		host="localhost",
# 		database="Data",
# 		user="postgres",
# 		password="admin"
# 	)
# df = pl.read_database(f"select state_t_30 from public.train where sample_id_int in ({', '.join(map(str, train_data.indices))})", connection=conn)
# plt.scatter(df, [0] * len(train_data.indices))
# plt.show()
# exit(0)
# # if __name__ == "__main__":
# # 	main()

# # exit(0)
# # b = pl.DataFrame({'a': [1]})
# # i = 8_750_000
# # while b.select(pl.len()).item() == 1:
# # 	b = pl.read_database(f"select sample_id from public.test where sample_id = 'train_{i}'", connection=conn)
# # 	print(i)
# # 	i += 10000
# # print(b, b.select(pl.len()).item())
# # print(i)
# # a = [pl.scan_parquet(f"Dataset/train/v1/train_{i}.parquet").select(pl.len()).collect().item() for i in range(51)]
# # s = 0
# # for i, v in enumerate(a):
# # 	s += v
# # 	print(i, s)
# # exit(0)
# # # b.write_database('test', connection='postgresql://postgres:admin@localhost:5432/Data', if_table_exists='append')

# # # import matplotlib.pyplot as plt
# # # a = pl.scan_parquet(f"Dataset/train/v1/train_15.parquet").slice(offset=150_000, length=1).collect()
# # 	# plt.plot(a['ptend_q0001_50'])
# # 	# plt.show()
# # exit(0)
# # file_rows = 100
# # left_to_do = file_rows
# # file_nr = 0
# # schema = pl.scan_parquet("Dataset/train/v1/train_0.parquet").drop('sample_id').schema
# # remainder = pl.DataFrame(schema=schema)
# # exit(0)
# # for i in trange(51, desc='files'):
# # 	ff = pl.scan_parquet(f"Dataset/train/v1/train_{i}.parquet").drop('sample_id')
# # 	l = ff.select(pl.len()).collect().item()
# # 	left_to_do = file_rows
# # 	last_offset = 0
# # 	while last_offset < l:
# # 		this_time = ff.slice(last_offset, file_rows)
# # 		last_offset += this_time.select(pl.len()).collect().item()

# # 		remainder = pl.concat([remainder, this_time.collect()]).write_parquet(f"Dataset/train/v1/train_{file_nr}.parquet")


# # last written: 3 + 9
# # last row: 8765944
# # a = pl.scan_parquet(f"Dataset/train/v1/train_43.parquet").slice(offset=8765945-8615945).collect()
# # a.write_database('test', connection='postgresql://postgres:admin@localhost:5432/Data', if_table_exists='append')
# # exit(0)
# # batch_size = 50_000

# # for i in trange(44, 51, desc='train files'):
# # 	df = pl.scan_parquet(f"Dataset/train/v1/train_{i}.parquet")
# # 	l = df.select(pl.len()).collect().item()
# # 	for batch_start in trange(0, l, batch_size, desc='batches'):
# # 		batch = df.slice(batch_start, batch_size).collect()
# # 		le = batch.select(pl.len()).item()
# # 		if le < batch_size:
# # 			print(le)

# # 		batch.write_database('test', connection='postgresql://postgres:admin@localhost:5432/Data', if_table_exists='append')