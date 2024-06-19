import polars as pl, numpy as np
import torch
import torch.utils
from torchmetrics.regression import R2Score
# from torch.utils.data import DataLoader, Dataset
# import json
# from model_def import model
from utils.preprocess import *
import sys, time
from tqdm import trange
sys.stdout.reconfigure(encoding='utf-8')

torch.set_default_dtype(torch.float64)

DEVICE = torch.device('cuda')

test = pl.scan_parquet("../impl/Dataset/test/test.parquet")
# test = preprocess_functions['standardisation']['norm'](test.to_numpy()) #.insert_column(0, test.to_series(0))
weights = pl.read_csv("../impl/Dataset/weights.csv").drop('sample_id').cast(pl.Float64)
out_schema=weights.schema
# weights = torch.tensor(weights.to_numpy())[0].to('cpu')
weights = weights.to_numpy()[0]
# print(weights)

def get_r2(model, norm):
	model.eval()
	nr_rows = 10_000
	valid = pl.read_parquet([f"../impl/Dataset/train/v1/train_{i}.parquet" for i in range(49, 51)], n_rows=nr_rows).drop('sample_id').to_numpy()

	rez = preprocess_functions[norm]['norm'](valid)
	ins, outs = rez[:, :in_len], rez[:, in_len:]
	# print("Read in")
	model.eval()

	r2score = R2Score(num_outputs=368, multioutput="raw_values").to(DEVICE)
	ins = torch.tensor(ins, device=DEVICE)
	outs= torch.tensor(outs, device=DEVICE)

	prediction = model(ins)

	r2 = r2score(prediction, outs)
	return r2

def get_pred(sample: pl.DataFrame, norm: str):
	sample = preprocess_functions[norm]['norm'](sample.to_numpy())
	sample = torch.tensor(sample, device=DEVICE)
	prediction = model(sample).cpu()
	prediction = preprocess_functions[norm]['denorm'](prediction)
	weighted = pl.DataFrame((prediction.numpy()[None, :] * weights).squeeze(), schema=out_schema)
	return weighted

batch_size = 75_000
norm = 'standardisation'


import dill
with torch.no_grad():
	model = dill.load(open('Models/resnet_parallel/model_checkpoint_batch_4_epoch_4.pickle', 'rb'), ignore=True)['model']
	model.to(DEVICE)
	print("Getting r2")
	r2 = get_r2(model, norm)
	print("Got r2")
	model.eval()
	# submission = submission.map_rows(predict)
	submission = pl.concat([get_pred(test.drop("sample_id").slice(i, batch_size).collect(), norm) for i in trange(0, 625_000, batch_size)])
	# predict the mean for variables with negative R2
	le = submission.select(pl.len()).item()
	# print(submission['ptend_q0002_12'])

	print("Replacing negative r2")
	for i in range(len(submission.columns)): # without sample_id
		if r2[i] < 0:
			print(out_vars[i])
			name = submission.columns[i]
			s = pl.Series(values=[(data_insights[name]["mean"] * weights[i]) for _ in range(le)], name=name)
			submission.replace_column(i, s)

	sample_ids = test.select(pl.col("sample_id")).collect()
	submission.insert_column(0, sample_ids)
	# print(submission['ptend_q0002_12'])
	submission.write_parquet('submission.parquet')

a = pl.scan_parquet("submission.parquet")
assert len(a.columns) == len(out_vars) + 1, "Columns missing"
assert all([c in out_vars + ['sample_id'] for c in a.columns]), "Columns names are not OK"
assert a.select(pl.len()).collect().item() == 625_000, "Less than 625_000 rows found"