import gmr, cloudpickle
from my_utils import *
import polars as pl
from tqdm import trange
from torchmetrics.regression import R2Score
from torch.nn import L1Loss
torch.set_default_dtype(torch.float64)
# valid = pl.read_parquet("Dataset/train/v1/train_40.parquet", n_rows=100_000)
# valid.drop("sample_id")
# valid_in = normalize_subset(valid, in_vars, method='none')
# valid_out = normalize_subset(valid, out_vars, method='none')
# valid_out = torch.tensor(valid_out.to_numpy())
dset = CustomSQLDataset(norm_method="none")
splits = get_splits()
print(splits)
trs = tdata.random_split(dset, splits, generator=torch.Generator().manual_seed(50))
weights = pl.read_csv("Dataset/weights.csv").drop('sample_id').cast(pl.Float64)
# out_schema=weights.schema
weights = weights.to_numpy()[0]
print("Read data")

batch_size = 1_000
sqldloader = DataLoader(trs[-1], num_workers=0,
						batch_sampler=tdata.BatchSampler(tdata.SequentialSampler(trs[-1]), batch_size=batch_size, drop_last=False), #
						collate_fn=identity)
valid_in, valid_out = next(iter(sqldloader))
# dlen = valid_in.shape[0]
# submission = weights.clear(dlen)
valid_out = valid_out * weights
model = cloudpickle.load(open("gmm_model.pickle", 'rb'))
prediction = torch.tensor(model.predict(
		np.array([i for i in range(len(in_vars))]), valid_in), dtype=torch.float64)
prediction = prediction * weights
# prediction = torch.tensor(np.concatenate([model.predict(
# 		np.array([i for i in range(len(in_vars))]), valid_in.slice(i, batch_size).to_numpy()
# 	) for i in trange(0, dlen, batch_size)]))


r2score = R2Score(num_outputs=368, multioutput="raw_values")
r2score_true = R2Score(num_outputs=368)
maescore = L1Loss()

r2 = r2score(prediction, valid_out)
mae = maescore(prediction, valid_out)
tru = r2score_true(prediction, valid_out)
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
			act=valid_out[0][idx],
			pred=prediction[0][idx],
			diff=valid_out[0][idx] - prediction[0][idx],
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
plt.show()
prediction = prediction.cpu()
valid_out = valid_out.cpu()
from mpl_toolkits.basemap import Basemap
while True:
	c = input("column name:")
	c = out_vars.index(c)
	plt.plot(prediction[:, c], label='pred')
	plt.plot(valid_out[:, c], label='actual')
	plt.legend()
	plt.title(out_vars[c])
	plt.show()