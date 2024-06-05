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
sys.stdout.reconfigure(encoding='utf-8')


in_vars = expand_vars(in_vars)
out_vars = expand_vars(out_vars)
zeroed_vars = expand_vars(zeroed_vars)
non_zeroed_out_vars = [v for v in out_vars if v not in zeroed_vars]
all_vars = ["sample_id"] + in_vars + out_vars

data_insights = json.load(open("data_insights.json"))
in_means = np.array([data_insights[v]["mean"] for v in in_vars])
out_means = np.array([data_insights[v]["mean"] for v in out_vars])
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

preprocess_functions = {
	"+mean/std": {"norm": preprocess_standardisation, "denorm": preprocess_destandardisation},
	"+mean": {"norm": preprocess_centered, "denorm": preprocess_decentered},
	"none": {"norm": preprocess_none, "denorm": preprocess_denone},
}

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
		df = pl.read_database(f"select * from public.train where sample_id_int = ({idx})", connection=conn).drop('sample_id_int')
		features, target = self.norm(df.to_numpy())
		conns.putconn(conn)
		return features.squeeze(), target.squeeze()
	def __getitems__(self, ids):
		global conns
		conn = conns.getconn()
		df = pl.read_database(f"select * from public.train where sample_id_int in ({', '.join(map(str, ids))})", connection=conn).drop('sample_id_int')
		features, target = self.norm(df.to_numpy())
		conns.putconn(conn)
		return features.squeeze(), target.squeeze()


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
	return torch.tensor(x[0], dtype=torch.float64, device=DEVICE), torch.tensor(x[1], dtype=torch.float64, device=DEVICE)


class SkipConnection(torch.nn.Module):
	def __init__(self, layers):
		super().__init__()
		self.actual = layers

	def forward(self, x):
		return torch.concat([self.actual(x), x], dim=1)


if __name__ == '__main__':
	BATCH_SIZE = 200
	sqldloader = DataLoader(CustomSQLDataset(), # batch_size=BATCH_SIZE, shuffle=True,
						 num_workers=4, prefetch_factor=5,# pin_memory=True, pin_memory_device='cuda',
						 batch_sampler=tdata.BatchSampler(tdata.RandomSampler(CustomSQLDataset()), batch_size=BATCH_SIZE, drop_last=False),
						 collate_fn=identity,)
						#  generator=torch.Generator(device='cuda'))
	# cc = conns.getconn()
	for xs, ys in enumerate(tqdm.tqdm(sqldloader)):
		pass
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