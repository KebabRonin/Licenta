import polars as pl, numpy as np
import torch
import torch.utils
from torchmetrics.regression import R2Score
# from torch.utils.data import DataLoader, Dataset
# import json
# from model_def import model
from my_utils import *
import sys, time
from tqdm import trange
from sklearn.metrics import r2_score
from model_def import model
sys.stdout.reconfigure(encoding='utf-8')

torch.set_default_dtype(torch.float64)

DEVICE = torch.device('cuda')

test = pl.read_parquet("Dataset/test/test.parquet")
sample_ids = test.drop_in_place("sample_id")
test.drop("sample_id")
test = normalize_subset(test) #.insert_column(0, test.to_series(0))
weights = pl.read_csv("Dataset/weights.csv").drop('sample_id').cast(pl.Float64)
dlen = test.select(pl.len()).item()
print(dlen)
out_schema=weights.schema
# weights = torch.tensor(weights.to_numpy())[0].to('cpu')
weights = weights.to_numpy()[0]
# print(weights)

def get_r2(model):
	model.eval()
	nr_rows = 10_000
	valid = pl.read_parquet([f"Dataset/train/v1/train_{i}.parquet" for i in range(49, 51)], n_rows=nr_rows)

	ins = normalize_subset(valid,in_vars, method="+mean/std")
	outs = normalize_subset(valid,out_vars, method="+mean/std")
	# print("Read in")
	model.eval()

	r2score = R2Score(num_outputs=368, multioutput="raw_values").to(DEVICE)
	ins = torch.tensor(ins.to_numpy(), device=DEVICE)
	outs= torch.tensor(outs.to_numpy(), device=DEVICE)

	prediction = model(ins)

	r2 = r2score(prediction, outs)
	return r2

def predict(name):
	sample = test.row(by_predicate=pl.col('sample_id').eq(name[0]))
	prediction = model(torch.tensor([sample[1:]]).to(DEVICE))
	prediction = (prediction * weights).to('cpu')
	weighted = tuple(([sample[0]] + prediction.tolist()))
	return weighted
def get_pred(sample: pl.DataFrame):
	sample = torch.tensor(sample.to_numpy(), device=DEVICE)
	prediction = model(sample)
	df = pl.DataFrame(preprocess_destandardisation(prediction.to('cpu').numpy()), schema=out_schema) # pred
	# undo standardisation
	# df = normalize_subset(pred, denormalize=True)
	# for i in range(27):
	# 	prediction[:, 120+i] = -sample[:, 120+i] / 1200
	# weight prediction (required by competition)
	weighted = pl.DataFrame((df.to_numpy()[None, :] * weights).squeeze(), schema=out_schema)
	return weighted

batch_size = 25_000

with torch.no_grad():
	model = torch.load('model_resnet5_worse.pt')
	model.to(DEVICE)
	r2 = get_r2(model)
	model.eval()
	# submission = submission.map_rows(predict)
	submission = pl.concat([get_pred(test.slice(i, batch_size)) for i in trange(0, dlen, batch_size)])
	# predict the mean for variables with negative R2
	le = submission.select(pl.len()).item()
	# print(submission['ptend_q0002_12'])

	for i in range(len(submission.columns)): # without sample_id
		if r2[i] <= 0:
			print(out_vars[i])
			name = submission.columns[i]
			s = pl.Series(values=[(data_insights[name]["mean"] * weights[i]) for _ in range(le)], name=name)
			submission.replace_column(i, s)

	submission.insert_column(0, sample_ids)
	# print(submission['ptend_q0002_12'])
	submission.write_parquet('submission.parquet')

a = pl.scan_parquet("submission.parquet")
assert len(a.columns) == len(out_vars) + 1, "Columns missing"
assert all([c in out_vars + ['sample_id'] for c in a.columns]), "Columns names are not OK"
assert a.select(pl.len()).collect().item() == 625_000, "Less than 625_000 rows found"