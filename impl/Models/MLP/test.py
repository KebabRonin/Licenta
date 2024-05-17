import polars as pl, numpy as np, torch
# from torcheval.metrics.functional import r2_score
from torchmetrics.regression import R2Score
from torch.nn import L1Loss
# from model_def import model
from my_utils import *
import sys, time, json
sys.stdout.reconfigure(encoding='utf-8')

model = torch.load('model.pt')
data_insights = json.load(open('data_insights.json'))
nr_rows = 10_000
valid = pl.read_parquet([f"Dataset/train/v1/train_{i}.parquet" for i in range(49, 51)], n_rows=nr_rows)

ins = normalize_subset(valid,in_vars, method="+mean/std")
outs = normalize_subset(valid,out_vars, method="+mean/std")
# print("Read in")
model.eval()

r2score = R2Score(num_outputs=368, multioutput="raw_values").to('cuda')
maescore = L1Loss()

print(f"Tested at: {time.strftime('%d-%b-%Y | %H:%M:%S')}")
print(model)
with torch.no_grad():
	ins = torch.tensor(ins.to_numpy(), device='cuda')
	outs= torch.tensor(outs.to_numpy(), device='cuda')
	# for sample, a in (zip(ins.iter_rows(), outs.iter_rows())):
	prediction = model(ins)
	# print(prediction[0][:20])
	# print(outs[0][:20])
	r2 = r2score(prediction, outs)
	mae = maescore(prediction, outs)
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
				act=outs[0][idx],
				pred=prediction[0][idx],
				diff=outs[0][idx] - prediction[0][idx],
				r2ll=r2[idx],
			) for idx in range(i, i+3) if idx < len(out_vars)
		), sep='')
	print(f"{'&':=^186}")
	print("MAE      :", mae.item())
	nice= sum(list(filter(lambda x: x >= 0, r2.tolist())))
	notnice= sum(r2.tolist())
	print("*Nice* r2:", nice/len(outs))
	print("Actual r2:", notnice/len(outs))
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