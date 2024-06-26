import gmr, cloudpickle
from my_utils import *
import polars as pl, numpy as np
from tqdm import trange
from torchmetrics.regression import R2Score
from torch.nn import L1Loss
torch.set_default_dtype(torch.float64)
# valid = pl.read_parquet("Dataset/train/v1/train_40.parquet", n_rows=100_000)
# valid.drop("sample_id")
# valid_in = normalize_subset(valid, in_vars, method='none')
# valid_out = normalize_subset(valid, out_vars, method='none')
# valid_out = torch.tensor(valid_out.to_numpy())
dset = TimestepSQLDataset(norm_method="none")
splits = get_splits(0.02)
print(splits)
trs = tdata.random_split(dset, splits, generator=torch.Generator().manual_seed(0))
weights = pl.read_csv("Dataset/weights.csv").drop('sample_id').cast(pl.Float64)
# out_schema=weights.schema
weights = weights.to_numpy()[0]

batch_size = 50
sqldloader = DataLoader(trs[-5], num_workers=0,
						batch_sampler=tdata.BatchSampler(tdata.RandomSampler(trs[-5], generator=torch.Generator().manual_seed(0)), batch_size=batch_size, drop_last=False), #
						collate_fn=identity)
valid_in, valid_out = next(iter(sqldloader))
print("Read data")
# dlen = valid_in.shape[0]
# submission = weights.clear(dlen)
model = cloudpickle.load(open("gmm_model.pickle", 'rb'))
prediction = torch.tensor(model.predict(
		np.array([i for i in range(len(in_vars))]), valid_in), dtype=torch.float64)
# prediction = torch.tensor(np.concatenate([model.predict(
# 		np.array([i for i in range(len(in_vars))]), valid_in.slice(i, batch_size).to_numpy()
# 	) for i in trange(0, dlen, batch_size)]))

print(prediction.dtype, valid_out.dtype)

r2score = R2Score(num_outputs=368, multioutput="raw_values")
r2score_true = R2Score(num_outputs=368)
maescore = L1Loss()

_, v_out = preprocess_standardisation(np.concatenate([valid_in.numpy(), valid_out.numpy()], axis=1))
_, pred = preprocess_standardisation(np.concatenate([valid_in.numpy(), prediction.numpy()], axis=1))
v_out, pred = torch.tensor(v_out, dtype=torch.float64), torch.tensor(pred, dtype=torch.float64)
r2 = r2score(pred, v_out)
mae = maescore(pred, v_out)
tru = r2score_true(pred, v_out)
# print(f"{r2=}")
# print(f"{mae=}")
header=f"{'Name':^15}|{'Actual':^10}|{'Pred':^10}|{'Diff':^10}|{'R2':^10}|*|"
# ll = 3*len(header)
# print(ll)
print(f"{'&':=^186}")
print(*(header for _ in range(3)), sep='')
print(f"{'&':=^186}")
fstr = "{vname:<15}|{act:>10.5f}|{pred:>10.5f}|{diff:>10.5f}|{r2ll:>10.5f}|*|"
for i in range(0, len(out_vars), 3):
	print(*(
		fstr.format(
			vname=out_vars[idx],
			act=v_out[0][idx],
			pred=pred[0][idx],
			diff=v_out[0][idx] - pred[0][idx],
			r2ll=r2[idx],
		) for idx in range(i, i+3) if idx < len(out_vars)
	), sep='')
print(f"{'&':=^186}")
mask = np.ones(len(r2), dtype=bool)
ok_indices = [idx for idx in range(len(out_vars)) if out_vars[idx] in zeroed_vars]
mask[ok_indices] = False
result = r2[mask,...]
nice= sum(result.tolist()) #list(filter(lambda x: x >= 0, result.tolist())))
nicer = sum(list(filter(lambda x: x >= 0, result.tolist())))
notnice= sum(r2.tolist())

print("MAE       :", mae.item())
print("*Nice*  r2:", nice/len(non_zeroed_out_vars))
print("*Nicer* r2:", nicer/len(non_zeroed_out_vars))
print("Actual  r2:", notnice/len(out_vars))
print("True", tru)
plt.plot(r2.cpu())
plt.xticks([0, 60, 120, 180, 240, 300, 360], ['ptend_t', 'ptend_q0001', 'ptend_q0002', 'ptend_q0003', 'ptend_u', 'ptend_v', 'cam_out'])
plt.axis((0, 368, -1, 1.2))
plt.grid()
plt.show()
pred = pred.cpu()
v_out = v_out.cpu()
from mpl_toolkits.basemap import Basemap
while True:
	c = input("column name:")
	c = out_vars.index(c)
	plt.plot(pred[:, c], label='pred')
	plt.plot(v_out[:, c], label='actual')
	plt.legend()
	plt.title(out_vars[c])
	plt.show()