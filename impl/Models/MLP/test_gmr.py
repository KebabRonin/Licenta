import gmr, cloudpickle
from my_utils import *
import polars as pl
from tqdm import trange
from torchmetrics.regression import R2Score
from torch.nn import L1Loss

valid = pl.read_parquet("Dataset/train/v1/train_40.parquet", n_rows=100_000)
valid.drop("sample_id")
valid_in = normalize_subset(valid, in_vars, method='none')
valid_out = normalize_subset(valid, out_vars, method='none')
valid_out = torch.tensor(valid_out.to_numpy())

weights = pl.read_csv("Dataset/weights.csv").drop('sample_id').cast(pl.Float64)
dlen = valid.select(pl.len()).item()
submission = weights.clear(dlen)
out_schema=weights.schema
weights = weights.to_numpy()[0]
print("Read data")

batch_size = 10_000
model = cloudpickle.load(open("gmm_model.pickle", 'rb'))
prediction = torch.tensor(np.concatenate([model.predict(
		np.array([i for i in range(len(in_vars))]), valid_in.slice(i, batch_size).to_numpy()
	) for i in trange(0, dlen, batch_size)]))


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