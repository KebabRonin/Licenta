import numpy as np, polars as pl
from my_utils import *
import gmr, cloudpickle, time, gc
from sklearn.mixture import GaussianMixture
from torchmetrics.regression import R2Score

n_samples = 100_000
n_samples_valid = 5_000
train = pl.scan_parquet("Dataset/train/v1/train_1.parquet", n_rows=n_samples).drop('sample_id').collect()
print("Read",  time.strftime('%H:%M:%S', time.localtime()))

loss_function = R2Score(num_outputs=368)
gmm = gmr.GMM(n_components=30, verbose=100)
train = train.to_numpy()
gmm.from_samples(train, n_iter=10)
del train
print("Trained",  time.strftime('%H:%M:%S', time.localtime()))
cloudpickle.dump(gmm, open("gmm_model.pickle", 'wb'))

gc.collect()

valid = pl.scan_parquet("Dataset/train/v1/train_49.parquet", n_rows=n_samples_valid).drop('sample_id').collect()
valid_in = normalize_subset(valid, in_vars, method='none').to_numpy()
valid_out = normalize_subset(valid, out_vars, method='none').to_numpy()
print("Read valid",  time.strftime('%H:%M:%S', time.localtime()))
Y = gmm.predict(np.array([i for i in range(valid_in.shape[1])]), valid_in)
print("Predicted",  time.strftime('%H:%M:%S', time.localtime()))

print(Y.shape)
print(loss_function(torch.tensor(valid_out), torch.tensor(Y)))
for i in range(Y.shape[1]):
	print(out_vars[i], Y[0][i], valid_out[0][i])
