import polars as pl, numpy as np
import torch
import torch.utils
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

test = pl.read_parquet("Dataset/test/test.parquet")
test = normalize_subset(test) #.insert_column(0, test.to_series(0))
weights = pl.read_csv("Dataset/submission_weights.csv").cast(pl.Float64)
dlen = test.select(pl.len()).item()
submission = weights.clear(dlen)
submission.replace_column(0, test['sample_id'])
out_schema=weights.drop('sample_id').schema
weights = torch.tensor(weights.drop("sample_id").to_numpy())[0].to('cuda')
# print(weights)

def predict(name):
	sample = test.row(by_predicate=pl.col('sample_id').eq(name[0]))
	prediction = model(torch.tensor([sample[1:]]).to('cuda'))
	prediction = (prediction * weights).to('cpu')
	weighted = tuple(([sample[0]] + prediction.tolist()))
	return weighted
def get_pred(sample: pl.DataFrame):
	# sample = test.slice(start, end)
	sample_ids = sample.to_series(0)
	sample = sample.drop("sample_id")
	# print(sample)
	prediction = model(torch.tensor(sample.to_numpy(), device='cuda'))
	# print(prediction[:5, :2])
	weighted = pl.DataFrame((prediction * weights).to('cpu').numpy(), schema=out_schema)
	df = weighted.insert_column(0, sample_ids) # weighted.with_columns(sample_id=sample_ids)
	return normalize_subset(df, denormalize=True)

batch_size = 125_000

with torch.no_grad():
	model = torch.load('model.pt')
	# submission = submission.map_rows(predict)
	submission = pl.concat([get_pred(test.slice(i, batch_size)) for i in trange(0, dlen, batch_size)])
	submission.write_parquet('submission.parquet')
