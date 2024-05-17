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
import numpy as np


in_vars = expand_vars(in_vars)
out_vars = expand_vars(out_vars)
zeroed_vars = expand_vars(zeroed_vars)
non_zeroed_out_vars = [v for v in out_vars if v not in zeroed_vars]
all_vars = ["sample_id"] + in_vars + out_vars

data_insights = json.load(open("data_insights.json"))
in_means = np.array([data_insights[v]["mean"] for v in in_vars])
out_means = np.array([data_insights[v]["mean"] for v in out_vars])
in_std_dev = np.array([data_insights[v]["std_dev"] for v in in_vars])
out_std_dev = np.array([data_insights[v]["std_dev"] for v in out_vars])

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
	plt.legend()
	plt.savefig(figname)

def normalize_subset(df:pl.DataFrame | pl.LazyFrame, columns=None, method="+mean/std", denormalize=False) -> pl.DataFrame | pl.LazyFrame:
	if columns is None:
		columns = df.columns
	if callable(method):
		return method(df, columns)
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


# import torch, polars as pl
# from torch.utils.data import Dataset
# from my_utils import in_vars, out_vars

# class MyDataset(Dataset):
# 	def __init__(self, in_files, normalize=False):
# 		self.lazy_df = pl.scan_parquet(in_files)
# 		self.normalize = normalize

# 	def __len__(self):
# 		return self.lazy_df.select(pl.len()).collect().item()

# 	def __getitem__(self, idx):
# 		row = self.lazy_df.slice(idx, 1).collect()
# 		x = torch.tensor(row.select(pl.col(in_vars)).to_numpy()).float()
# 		y = torch.tensor(row.select(pl.col(out_vars)).to_numpy()).float()
# 		return (x - in_means) / in_std_dev, (y - out_means) / out_std_dev