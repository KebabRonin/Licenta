import numpy as np, polars as pl
import utils.data, utils.dset
import torch.utils.data as tdata, tqdm, torch
import gmr, cloudpickle, time, gc, matplotlib.pyplot as plt
# from sklearn.mixture import GaussianMixture
from torchmetrics.regression import R2Score
# import pycave

n_samples = 200_000
n_samples_valid = 5_000
# train = pl.scan_parquet("Dataset/train/v1/train_1.parquet", n_rows=n_samples).drop('sample_id').collect()
# train = normalize_subset(train)
def train(batches):
	def we_all_become_one(x: tuple[np.ndarray, np.ndarray]):
		return np.concatenate([x[0], x[1]], dtype=np.float64, axis=1)

	gmm = gmr.GMM(n_components=500, verbose=100, random_state=0)

	for batch in tqdm.tqdm(batches, desc='batches', position=1):
		sqldloader = tdata.DataLoader(batch, num_workers=0,
								batch_sampler=tdata.BatchSampler(tdata.RandomSampler(batch), batch_size=n_samples, drop_last=False), #
								collate_fn=we_all_become_one)
		samples = next(iter(sqldloader))
		gmm.from_samples(samples, n_iter=40, init_params='random')
	print("Trained",  time.strftime('%H:%M:%S', time.localtime()))
	cloudpickle.dump(gmm, open("gmm_model2.pickle", 'wb'))

	gc.collect()
def pred(valid_in, valid_out):
	loss_function = R2Score(num_outputs=368, multioutput="raw_values")
	gmm = cloudpickle.load(open("gmm_model2.pickle", 'rb'))
	print("Read valid",  time.strftime('%H:%M:%S', time.localtime()))
	Y = gmm.predict(np.array([i for i in range(valid_in.shape[1])]), valid_in)

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
	splits = utils.dset.get_splits(norm_method="standardisation", dset_class=utils.dset.TimestepSQLDataset, fraction=0.02)
	# pred()
	train(splits[:1])
	pred(*next(iter(splits[-1])))