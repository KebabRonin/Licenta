in_vars = [
	("state_t", 60),
	("state_q0001", 60),
	("state_q0002", 60),
	("state_q0003", 60),
	("state_u", 60),
	("state_v", 60),
	("state_ps", 1),
	("pbuf_SOLIN", 1),
	("pbuf_LHFLX", 1),
	("pbuf_SHFLX", 1),
	("pbuf_TAUX", 1),
	("pbuf_TAUY", 1),
	("pbuf_COSZRS", 1),
	("cam_in_ALDIF", 1),
	("cam_in_ALDIR", 1),
	("cam_in_ASDIF", 1),
	("cam_in_ASDIR", 1),
	("cam_in_LWUP", 1),
	("cam_in_ICEFRAC", 1),
	("cam_in_LANDFRAC", 1),
	("cam_in_OCNFRAC", 1),
	("cam_in_SNOWHLAND", 1),
	("pbuf_ozone", 60),
	("pbuf_CH4", 60), # 27-59 dropped because constant (=)
	("pbuf_N2O", 60), # 27-59 dropped because constant (=)
]

out_vars = [
	("ptend_t", 60),
	("ptend_q0001", 60), # 0-11 zeroed by submission weights
	("ptend_q0002", 60), # 0-14 zeroed by submission weights
	("ptend_q0003", 60), # 0-11 zeroed by submission weights
	("ptend_u", 60),     # 0-11 zeroed by submission weights
	("ptend_v", 60),     # 0-11 zeroed by submission weights
	("cam_out_NETSW", 1),
	("cam_out_FLWDS", 1),
	("cam_out_PRECSC", 1),
	("cam_out_PRECC", 1),
	("cam_out_SOLS", 1),
	("cam_out_SOLL", 1),
	("cam_out_SOLSD", 1),
	("cam_out_SOLLD", 1),
]

zeroed_vars = [
	("ptend_q0001", 12), # 0-11 zeroed by submission weights
	("ptend_q0002", 15), # 0-14 zeroed by submission weights
	("ptend_q0003", 12), # 0-11 zeroed by submission weights
	("ptend_u", 12),     # 0-11 zeroed by submission weights
	("ptend_v", 12),     # 0-11 zeroed by submission weights
]

def expand_vars(vars: list[str]):
	return sum([[v[0]] if v[1] == 1 else [f"{v[0]}_{i}" for i in range(v[1])] for v in vars], start=[])

import polars as pl, json, matplotlib.pyplot as plt
import numpy as np, torch, sys
# sys.stdout.reconfigure(encoding='utf-8')


in_vars = expand_vars(in_vars)
out_vars = expand_vars(out_vars)
zeroed_vars = expand_vars(zeroed_vars)
non_zeroed_out_vars = [v for v in out_vars if v not in zeroed_vars]
all_vars = ["sample_id"] + in_vars + out_vars

data_insights = json.load(open("data_insights.json"))
in_means = np.array([data_insights[v]["mean"] for v in in_vars])
out_means = np.array([data_insights[v]["mean"] for v in out_vars])
in_mins = np.array([data_insights[v]["min"] for v in in_vars])
out_mins = np.array([data_insights[v]["min"] for v in out_vars])
in_maxs = np.array([data_insights[v]["max"] for v in in_vars])
out_maxs = np.array([data_insights[v]["max"] for v in out_vars])
in_mms = np.array([in_maxs[i] - in_mins[i] if in_maxs[i] - in_mins[i] != 0 else 1 for i in range(len(in_vars))])
out_mms = np.array([out_maxs[i] - out_mins[i] if out_maxs[i] - out_mins[i] != 0 else 1 for i in range(len(out_vars))])
in_std_dev  = np.array([data_insights[v]["std_dev"] if data_insights[v]["std_dev"] != 0 else 1 for v in in_vars])
out_std_dev = np.array([data_insights[v]["std_dev"] if data_insights[v]["std_dev"] != 0 else 1 for v in out_vars])

def print_box(*args): #, width=40):
	print(f"**{'&':=^42}**")
	for t in args:
		print(f"|| {t:<40} ||")
	print(f"**{'&':=^42}**")


def plott(init_err, finl_err, figname="fig"):
	# plt.gca().set_ylim([-40, 1])
	plt.close()
	# fig = plt.figure()
	_plt_indices = [i for i in range(len(finl_err))]
	plt.bar(_plt_indices, init_err[:len(finl_err)], alpha=0.5, label='init', color='orange')# color=list(map(lambda x: 'blue' if x > 0 else 'red', init_err[:len(finl_err)])))
	plt.bar(_plt_indices, finl_err, label='final', alpha=0.5, color='cyan')# color=list(map(lambda x: 'cyan' if x > 0 else 'orange', finl_err)))
	plt.axis((None, None, -1, 1))
	plt.legend()
	plt.savefig(figname)
def preprocess_standardisation(arr:np.ndarray):
	features = ((arr[:, :556] - in_means )[None, :] / in_std_dev ).squeeze()
	targets  = ((arr[:, 556:] - out_means)[None, :] / out_std_dev).squeeze()
	return features, targets
def preprocess_destandardisation(arr:np.ndarray):
	targets  = ((arr[None, :] * out_std_dev)[None, :] + out_means).squeeze()
	return targets
def preprocess_mean_normalisation(arr:np.ndarray):
	features = ((arr[:, :556] - in_means )[None, :] / in_mms ).squeeze()
	targets  = ((arr[:, 556:] - out_means)[None, :] / out_mms).squeeze()
	return features, targets
def preprocess_mean_denormalisation(arr:np.ndarray):
	targets  = ((arr[None, :] * out_mms)[None, :] + out_means).squeeze()
	return targets
def preprocess_standardisation_minmax(arr:np.ndarray, a, b):
	# a, b = -10, 10
	features = a + (((arr[:, :556] - in_mins )[None, :] * (b-a)) / in_mms ).squeeze()
	targets  = a + (((arr[:, 556:] - out_mins)[None, :] * (b-a)) / out_mms).squeeze()
	return features, targets
def preprocess_destandardisation_minmax(arr:np.ndarray, a, b):
	# a, b = -10, 10
	targets  = (((arr[None, :] - a) * out_mms)[None, :] / (b-a) + out_mins).squeeze()
	return targets
def preprocess_centered(arr:np.ndarray):
	features = ((arr[:, :556] - in_means )).squeeze()
	targets  = ((arr[:, 556:] - out_means)).squeeze()
	return features, targets
def preprocess_decentered(arr:np.ndarray):
	targets  = (arr[None, :] + out_means).squeeze()
	return targets
def preprocess_none(arr:np.ndarray):
	features = (arr[:, :556]).squeeze()
	targets  = (arr[:, 556:]).squeeze()
	return features, targets
def preprocess_denone(arr:np.ndarray):
	targets  = (arr[:, :]).squeeze()
	return targets
def mmax10(x):
	return preprocess_standardisation_minmax(x, -10, 10)
def demmax10(x):
	return preprocess_destandardisation_minmax(x, -10, 10)
def mmax100(x):
	return preprocess_standardisation_minmax(x, -100, 100)
def demmax100(x):
	return preprocess_destandardisation_minmax(x, -100, 100)
preprocess_functions = {
	"mean norm": {"norm": preprocess_mean_normalisation, "denorm": preprocess_mean_denormalisation},
	"minmax10": {"norm": mmax10, "denorm": demmax10},
	"minmax100": {"norm": mmax100, "denorm": demmax100},
	"+mean/std": {"norm": preprocess_standardisation, "denorm": preprocess_destandardisation},
	"+mean": {"norm": preprocess_centered, "denorm": preprocess_decentered},
	"none": {"norm": preprocess_none, "denorm": preprocess_denone},
}
VALID_RESERVE=2
def get_splits(fraction = 0.02):
	splits = [fraction for _ in range(int(1//fraction))]
	splits += [1 - sum(splits)]
	return splits
# legacy
def normalize_subset(df:pl.DataFrame | pl.LazyFrame, columns=None, method="+mean/std", denormalize=False) -> pl.DataFrame | pl.LazyFrame:
	if columns is None:
		columns = df.columns
	if callable(method):
		return method(df, columns)
	if type(df) is pl.DataFrame or pl.LazyFrame:
		match method:
			case "+mean/std":
				if denormalize:
					return df.select(
							(pl.col(col) * (data_insights[col]['std_dev'] if data_insights[col]['std_dev'] != 0 else 1)
							+ data_insights[col]['mean']) if col in data_insights else pl.col(col)
							for col in columns)
				else:
					return df.select(
							((pl.col(col) -
							data_insights[col]['mean']) / (data_insights[col]['std_dev'] if data_insights[col]['std_dev'] != 0 else 1))
							if col in data_insights else pl.col(col)
							for col in columns)
			case "+mean":
				if denormalize:
					return df.select(
							(pl.col(col) +
							data_insights[col]['mean'])  if col in data_insights else pl.col(col)
							for col in columns)
				else:
					return df.select(
							(pl.col(col) -
							data_insights[col]['mean'])  if col in data_insights else pl.col(col)
							for col in columns)
			case "none":
				return df.select(
						(pl.col(col)
						for col in columns))
			case _:
				raise Exception("'method' not recognized. Must be callable or one of ['+mean/std', '+mean', 'none']")

from torch.utils.data import Dataset, DataLoader
import torch.utils.data as tdata
import psycopg2.pool as psycopg2_pool
conns = psycopg2_pool.ThreadedConnectionPool(1, 10, dbname="Data", user="postgres", password="admin", host="localhost")


class TimestepSQLDataset(Dataset):
	"""One sample is of size 384, which is one timestep over all map locations

	Args:
		norm_method (str): Normalization method, one of preprocess_functions.keys()
	"""
	def __init__(self, norm_method):
		self.norm_method = norm_method
		self.norm = preprocess_functions[norm_method]['norm']
		self.denorm = preprocess_functions[norm_method]['denorm']

	def __len__(self):
		return int(10_091_520 // 384) # 26_280

	def __getitem__(self, idx):
		global conns
		conn = conns.getconn()

		df = pl.read_database(f"select * from public.train where {idx*384} <= sample_id_int and sample_id_int < {(idx+1)*384} order by sample_id_int;", connection=conn)
		df = df.drop('sample_id_int')
		rez = self.norm(df.to_numpy())

		conns.putconn(conn)
		return rez #rez[:, :556], rez[:, 556:]
	def __getitems__(self, ids):
		ls = [self.__getitem__(i) for i in ids]
		return np.concatenate([l[0] for l in ls], axis=0), np.concatenate([l[1] for l in ls], axis=0)


class CustomSQLDataset(Dataset):
	def __init__(self, norm_method = "+mean/std"):
		self.norm_method = norm_method
		self.norm = preprocess_functions[norm_method]['norm']
		self.denorm = preprocess_functions[norm_method]['denorm']

	def __len__(self):
		return 10_091_520

	def __getitem__(self, idx):
		global conns
		conn = conns.getconn()
		df = pl.read_database(f"select * from public.train where sample_id_int = ({idx})", connection=conn)
		ids = df.drop_in_place('sample_id_int').to_numpy()
		df = df.drop('sample_id_int')
		features, target = self.norm(df.to_numpy())
		conns.putconn(conn)
		return features.squeeze(), target.squeeze()#, ids.squeeze()
	def __getitems__(self, ids):
		global conns
		conn = conns.getconn()
		df = pl.read_database(f"select * from public.train where sample_id_int in ({', '.join(map(str, ids))})", connection=conn)
		ids = df.drop_in_place('sample_id_int').to_numpy()
		df = df.drop('sample_id_int')
		features, target = self.norm(df.to_numpy())
		conns.putconn(conn)
		return features.squeeze(), target.squeeze()#, ids.squeeze()


import time
torch.set_default_dtype(torch.float64)
DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
# torch.set_default_device(DEVICE)

# sampl = iter(tdata.RandomSampler(CustomSQLDataset()))
# ids = [next(sampl) for _ in range(BATCH_SIZE)]

# for e in range(10):
# 	print([next(sampl) for _ in range(BATCH_SIZE)])
# 	if e > 10:
# 		break
# exit(0)
import tqdm

def identity(x: tuple[np.ndarray, np.ndarray]):
	return torch.tensor(x[0]), torch.tensor(x[1])

class SkipConnection(torch.nn.Module):
	def __init__(self, layers):
		super().__init__()
		self.actual = layers

	def forward(self, x):
		return torch.concat([self.actual(x), x], dim=1)


class ParallelModuleList(torch.nn.Module):
	def __init__(self, models):
		super().__init__()
		self.models = models

	def forward(self, x):
		out = torch.concat([layer(x) for layer in (self.models)], dim=1)
		return out


if __name__ == '__main__':
	# test norms
	data = pl.read_parquet('Dataset/train/v1/train_45.parquet', n_rows=2_000).drop('sample_id')
	dd = data.to_numpy()
	print(dd.shape)
	# print(dd[:20, None])
	for k in (preprocess_functions.keys()):
		d = preprocess_functions[k]['norm'](dd)
		tt = preprocess_functions[k]['denorm'](d[1])
		print(k, np.sum(np.abs(dd[:, 556:] - tt)))
		# print(dd[:, 556:][:5, 0])
		# print(tt[:5, 0])

	BATCH_SIZE = 200
	dset = CustomSQLDataset(norm_method='none')
	sqldloader = DataLoader(dset, # batch_size=BATCH_SIZE, shuffle=True,
						 num_workers=4, prefetch_factor=5,# pin_memory=True, pin_memory_device='cuda',
						 batch_sampler=tdata.BatchSampler(tdata.RandomSampler(dset, generator=torch.Generator().manual_seed(0)), batch_size=BATCH_SIZE, drop_last=False),
						 collate_fn=identity,)
						#  generator=torch.Generator(device='cuda'))
	# cc = conns.getconn()
	# for xs, ys in enumerate(tqdm.tqdm(sqldloader)):
	# 	pass
	# ids = [random.randrange(0, 10_091_520) for i in range(BATCH_SIZE)]
	t0 = time.time()
	# sampl = iter(tdata.RandomSampler(CustomSQLDataset()))
	# ids = [str(next(sampl)) for _ in range(BATCH_SIZE)]
	# df = pl.read_database(f"select * from public.train where sample_id_int in ({', '.join(ids)})", connection=cc).drop('sample_id_int')
	# df = df.to_numpy().squeeze()
	# xs, ys = preprocess_functions["+mean/std"]['norm'](df)
	xs, ys = next(iter(sqldloader))
	print(time.time() - t0)
	print(xs.shape, ys.shape, xs.dtype, ys.dtype)
	print(xs)