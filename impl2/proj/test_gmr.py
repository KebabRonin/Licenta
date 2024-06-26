import gmr, cloudpickle
from utils.dset import *
from utils.nn import identity
from utils.data import *
from utils.preprocess import *
from torch.utils.data import DataLoader
import torch.utils.data as tdata, torch
import polars as pl, numpy as np
from tqdm import trange
from torchmetrics.regression import R2Score
import matplotlib.pyplot as plt
from torch.nn import L1Loss
torch.set_default_dtype(torch.float64)
# valid = pl.read_parquet("Dataset/train/v1/train_40.parquet", n_rows=100_000)
# valid.drop("sample_id")
# valid_in = normalize_subset(valid, in_vars, method='none')
# valid_out = normalize_subset(valid, out_vars, method='none')
# valid_out = torch.tensor(valid_out.to_numpy())
splits = get_splits('none', SQLDataset, 0.02)


r2score = R2Score(num_outputs=368, multioutput="raw_values")
r2score_true = R2Score(num_outputs=368)
print("Read data")
# dlen = valid_in.shape[0]
# submission = weights.clear(dlen)
import dill
vsc = dill.load(open('valid_scores.dill', 'rb'))['gmm'][0]
r2s = torch.zeros(368)
preds = []
for i in trange(0, 50, 10):
	# cupy.clear_memo
	valid_in, valid_out = splits[-4].__getitems__(range(i*384, (i+10)*384))
	model = cloudpickle.load(open("../impl/gmm_model.pickle", 'rb'))
	prediction = model.predict(
			np.array([i for i in range(len(in_vars))]), valid_in)

	v_out = preprocess_standardisation(valid_out)
	pred = preprocess_standardisation(prediction)
	# for j in range(368):
	# 	if vsc[j] <= 0:
	# 		pred[:, j] = 0
	v_out, pred = torch.tensor(v_out, dtype=torch.float64), torch.tensor(pred, dtype=torch.float64)
	preds.append(pred)
	# r2 = r2score(pred, v_out)
	# r2s += r2.cpu()
# r2s /= 5
# r2s = r2s.numpy()
pred = torch.concatenate(preds, dim=0)
print(pred.shape)
dill.dump(pred, open('rezs/valid2/gmm.dill', 'wb'))
exit(0)

import dill
dill.dump(r2s, open('gmr_r2.dill', 'wb'))
plt.plot(r2s)
plt.xticks([0, 60, 120, 180, 240, 300, 360], ['ptend_t', 'ptend_q0001', 'ptend_q0002', 'ptend_q0003', 'ptend_u', 'ptend_v', 'cam_out'])
plt.axis((0, 368, -1, 1.2))
plt.grid()
plt.show()

# pred = pred.cpu()
# v_out = v_out.cpu()
# from mpl_toolkits.basemap import Basemap
# while True:
# 	c = input("column name:")
# 	c = out_vars.index(c)
# 	plt.plot(pred[:, c], label='pred')
# 	plt.plot(v_out[:, c], label='actual')
# 	plt.legend()
# 	plt.title(out_vars[c])
# 	plt.show()