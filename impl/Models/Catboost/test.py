import polars as pl, torch
# from torcheval.metrics.functional import r2_score
from torchmetrics.regression import R2Score
from my_utils import in_vars, out_vars, normalize_subset, non_zeroed_out_vars
import sys, json
import catboost, tqdm
sys.stdout.reconfigure(encoding='utf-8')

# model = torch.load('model.pt')
nr_rows = 10_000
valid = pl.read_parquet([f"Dataset/train/v1/train_{i}.parquet" for i in range(49, 51)], n_rows=nr_rows)

ins = normalize_subset(valid, in_vars).to_numpy()
outs = normalize_subset(valid, out_vars)
print("Read in")

model = catboost.CatBoostRegressor()

r2score = R2Score(num_outputs=1).to('cuda') #, multioutput="raw_values"

r2tot = 0
nrvars = 0
# for sample, a in (zip(ins.iter_rows(), outs.iter_rows())):
scores = []
try:
	for var in tqdm.tqdm(out_vars):
		if var not in non_zeroed_out_vars:
			scores.append(1)
		else:
			try:
				model.load_model(f"CatBoostModel/{var}_model.cbm")
				outs_var = torch.tensor(outs.select(pl.col(var)).to_numpy().squeeze())
				prediction = torch.tensor(model.predict(ins))
				# print(prediction)
				# print(outs_var)
				r2 = r2score(prediction, outs_var)
				scores.append(r2.cpu().item())
				nrvars+= 1
				# print(var, ":", r2.item())
				if r2 > 0:
					r2tot += r2
			except Exception as e:
				print(var, e)
	print("Done")
	print(nrvars)
	print(r2tot/nrvars)
except Exception as e:
	print(e)
	print(nrvars)
	print(r2tot/nrvars)

import matplotlib.pyplot as plt
plt.plot(scores)
plt.show()

# l = valid.select(pl.len()).collect().item()
# each = int(l/20)
# for i in valid.iter_slices(each):
# 	df = valid.slice(i*each, each).cast({v:pl.Float32 for v in (in_vars + out_vars)})
# 	print(df.slice(0, 1).collect())
# 	df.collect()
# 	df.write_parquet("Dataset/train/v3/train_{i}.parquet")