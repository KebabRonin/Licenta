import polars as pl, numpy as np, torch
# from torcheval.metrics.functional import r2_score
from torchmetrics.regression import R2Score
from model_def import model
from my_utils import in_vars, out_vars
import sys, time, json
from tqdm import tqdm
sys.stdout.reconfigure(encoding='utf-8')

model = torch.load('model.pt')
data_insights = json.load(open('data_insights.json'))
# nr_rows = 100_000
valid = pl.read_parquet([f"Dataset/train/v1/train_{i}.parquet" for i in range(49, 51)])

ins = valid.select(
	(pl.col(in_vars[i]) -
	data_insights[in_vars[i]]['mean']) #/ (data_insights[in_vars[i]]['std_dev'] if data_insights[in_vars[i]]['std_dev'] != 0 else 1)
	for i in range(len(in_vars))
)
outs = valid.select(
	(pl.col(out_vars[i]) -
	data_insights[out_vars[i]]['mean']) #/ (data_insights[out_vars[i]]['std_dev'] if data_insights[out_vars[i]]['std_dev'] != 0 else 1)
	for i in range(len(out_vars))
)

model.eval()

r2score = R2Score(num_outputs=368, multioutput="raw_values").to('cuda')

with torch.no_grad():
	ins = torch.tensor(ins.to_numpy(), device='cuda')
	outs= torch.tensor(outs.to_numpy(), device='cuda')
	# for sample, a in (zip(ins.iter_rows(), outs.iter_rows())):
	prediction = model(ins)
	print(prediction[0][:20])
	print(outs[0][:20])
	r2 = r2score(prediction, outs)
	print(r2)
	print(sum(r2))
# import polars as pl
# import sys, json
# from my_utils import in_vars, out_vars
# sys.stdout.reconfigure(encoding='utf-8')

# data_insights = json.load(open('data_insights.json'))
# valid = pl.scan_parquet([f"Dataset/train/v1/train_{i}.parquet" for i in range(51)])

# l = valid.select(pl.len()).collect().item()
# each = int(l/20)
# for i in valid.iter_slices(each):
# 	df = valid.slice(i*each, each).cast({v:pl.Float32 for v in (in_vars + out_vars)})
# 	print(df.slice(0, 1).collect())
# 	df.collect()
# 	df.write_parquet("Dataset/train/v3/train_{i}.parquet")