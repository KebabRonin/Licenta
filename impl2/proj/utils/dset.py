from utils.preprocess import preprocess_functions
from utils.data import in_len
from torch.utils.data import Dataset
import polars as pl, numpy as np
import psycopg2.pool as psycopg2_pool

# not part of base class because it can't be pickled
conns = psycopg2_pool.ThreadedConnectionPool(1, 5, dbname="Data", user="postgres", password="admin", host="localhost")

class SQLDataset(Dataset):
	def __init__(self, norm_method):
		self.norm_method = norm_method
		self.norm = preprocess_functions[norm_method]['norm']
		self.denorm = preprocess_functions[norm_method]['denorm']

	def __len__(self):
		return 10_091_520

	# this is very inefficient
	# def __getitem__(self, idx):
	# 	global conns
	# 	conn = conns.getconn()
	# 	df = pl.read_database(f"select * from public.train where sample_id_int = ({idx})", connection=conn)
	# 	ids = df.drop_in_place('sample_id_int').to_numpy()
	# 	df = df.drop('sample_id_int')
	# 	rez = self.norm(df.to_numpy())
	# 	conns.putconn(conn)
	# 	return rez[:, :in_len], rez[:, in_len:]
	def __getitems__(self, ids):
		global conns
		conn = conns.getconn()
		df = pl.read_database(f"select * from public.train where sample_id_int in ({', '.join(map(str, ids))})", connection=conn)
		# ids = df.drop_in_place('sample_id_int').to_numpy()
		df = df.drop('sample_id_int')
		rez = self.norm(df.to_numpy())
		conns.putconn(conn)
		return rez[:, :in_len], rez[:, in_len:]

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
		if not isinstance(idx, int):
			return self.__getitems__(idx)
		global conns
		conn = conns.getconn()

		df = pl.read_database(f"select * from public.train where {idx*384} <= sample_id_int and sample_id_int < {(idx+1)*384} order by sample_id_int;", connection=conn)
		df = df.drop('sample_id_int')
		rez = self.norm(df.to_numpy())

		conns.putconn(conn)
		return rez[:, :in_len], rez[:, in_len:]
	def __getitems__(self, ids):
		ls = [self.__getitem__(i) for i in ids]
		return np.concatenate([l[0] for l in ls], axis=0), np.concatenate([l[1] for l in ls], axis=0)

import torch, torch.utils.data as tdata

VALID_RESERVE=2
def get_splits(norm_method, dset_class=SQLDataset, fraction=0.02):
	dset = dset_class(norm_method)
	splits = [fraction for _ in range(int(1//fraction))]
	splits += [1 - sum(splits)]
	print("Batch splits:", splits)
	trs = tdata.random_split(dset, splits, generator=torch.Generator().manual_seed(0))
	return trs