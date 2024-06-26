import numpy as np, polars as pl
from my_utils import *
import gmr, cloudpickle, time, gc, matplotlib.pyplot as plt
# from sklearn.mixture import GaussianMixture
from torchmetrics.regression import R2Score
# import pycave

n_samples = 200_000
n_samples_valid = 500
# train = pl.scan_parquet("Dataset/train/v1/train_1.parquet", n_rows=n_samples).drop('sample_id').collect()
# train = normalize_subset(train)
def train():
	batch = trs[0]
	def we_all_become_one(x: tuple[np.ndarray, np.ndarray]):
		return np.concatenate([x[0], x[1]], dtype=np.float64, axis=1)

	sqldloader = DataLoader(batch, num_workers=0,
							batch_sampler=tdata.BatchSampler(tdata.RandomSampler(batch), batch_size=n_samples, drop_last=False), #
							collate_fn=we_all_become_one)
	train = next(iter(sqldloader))
	# exit(0)
	# conn = conns.getconn()
	# train = pl.read_database(f"select * from public.train where sample_id_int in ({', '.join(map(str, ids))}) limit {n_samples};", connection=conn).drop('sample_id_int')
	# conns.putconn(conn)
	print("Read", time.strftime('%H:%M:%S', time.localtime()), '\n\n\n')
	print(type(train))
	gmm = gmr.GMM(n_components=200, verbose=100, random_state=0)
	# train = train.to_numpy()
	gmm.from_samples(train, n_iter=40, init_params='kmeans++')
	del train
	print("Trained",  time.strftime('%H:%M:%S', time.localtime()))
	cloudpickle.dump(gmm, open("gmm_model2.pickle", 'wb'))

	gc.collect()
def pred():
	loss_function = R2Score(num_outputs=368, multioutput="raw_values")
	gmm = cloudpickle.load(open("gmm_model.pickle", 'rb'))
	sqldloader = DataLoader(trs[-1], num_workers=0,
							batch_sampler=tdata.BatchSampler(tdata.SequentialSampler(trs[-1]), batch_size=n_samples_valid, drop_last=False), #
							collate_fn=identity)
	valid_in, valid_out = next(iter(sqldloader))
	# valid = pl.scan_parquet("Dataset/train/v1/train_49.parquet", n_rows=n_samples_valid).drop('sample_id').collect()
	# valid_in = normalize_subset(valid, in_vars).to_numpy()
	# valid_out = normalize_subset(valid, out_vars).to_numpy()
	print("Read valid",  time.strftime('%H:%M:%S', time.localtime()))
	T = 0
	trials = 5
	for _ in tqdm.trange(trials):
		t0 = time.time()
		Y = gmm.predict(np.array([i for i in range(valid_in.shape[1])]), valid_in)
		t1 = time.time() - t0
		# print("Predicted", t1,  time.strftime('%H:%M:%S', time.localtime()))
		T += t1
	print(T/trials)

	print(Y.shape)
	ll = loss_function(valid_out, torch.tensor(Y))
	print("R2:", ll.mean())
	# plt.plot(ll)
	# plt.xticks([0, 60, 120, 180, 240, 300, 360], ['ptend_t', 'ptend_q0001', 'ptend_q0002', 'ptend_q0003', 'ptend_u', 'ptend_v', 'cam_out'])
	# plt.grid()
	# plt.show()
	# for i in range(Y.shape[0]):
	# 	plt.plot(Y[i, :], label='pred')
	# 	plt.plot(valid_out[i, :], label='true')
	# 	plt.legend()
	# 	plt.show()
	# 	print(out_vars[i], Y[0][i], valid_out[0][i])

if __name__ == '__main__':
	dset = CustomSQLDataset(norm_method="none")
	splits = get_splits()
	trs = tdata.random_split(dset, splits, generator=torch.Generator().manual_seed(50))
	# pred()
	train()