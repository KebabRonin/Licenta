import polars as pl, torch
# from torcheval.metrics.functional import r2_score
from torchmetrics.regression import R2Score
import utils.dset as ds, utils.nn as nns
from utils.data import *
import sys, json, glob
import catboost, tqdm
sys.stdout.reconfigure(encoding='utf-8')
torch.set_default_dtype(torch.float64)
# model = torch.load('model.pt')
ddddd = ds.get_splits('standardisation', dset_class=ds.SQLDataset, fraction=0.02)
print(len(ddddd))
ins, outs = ddddd[-4].__getitems__(range(19_200))
model = catboost.CatBoostRegressor()

r2score = R2Score(num_outputs=1).to('cuda') #, multioutput="raw_values"

r2tot = 0
nrvars = 0
# for sample, a in (zip(ins.iter_rows(), outs.iter_rows())):
scores = []
preds = []
for var in tqdm.tqdm(out_vars):
	print(var, list(filter(lambda x: f'{var}_' in x, glob.glob("../impl/CatBoostModel7/*_model.cbm"))))
	if var in ['ptend_q0002_17','ptend_q0002_18'] or len(list(filter(lambda x: f'{var}_' in x, glob.glob("../impl/CatBoostModel7/*_model.cbm")))) == 0:
		outs_var = torch.tensor(outs[:, out_vars.index(var)].squeeze())
		prediction = torch.zeros_like(outs_var)
		preds.append(prediction)
		# r2 = r2score(prediction, outs_var)
		# scores.append(r2.cpu().item())
	else:
		model.load_model(f"../impl/CatBoostModel7/{var}_model.cbm")
		outs_var = torch.tensor(outs[:, out_vars.index(var)].squeeze())
		prediction = torch.tensor(model.predict(ins))
		preds.append(prediction)
		# print(torch.stack(preds, dim=1).shape)
		# print(prediction)
		# print(outs_var)
		# r2 = r2score(prediction, outs_var)
		# scores.append(r2.cpu().item())
		# nrvars+= 1
print("Done")
import dill
dill.dump(torch.stack(preds, dim=1), open('rezs/valid2/cb7.dill', 'wb'))

# import matplotlib.pyplot as plt, dill
# dill.dump(scores, open('cb8scores.dill', 'wb'))
# plt.plot(scores)
# plt.xticks([0, 60, 120, 180, 240, 300, 360], ['ptend_t', 'ptend_q0001', 'ptend_q0002', 'ptend_q0003', 'ptend_u', 'ptend_v', 'cam_out'])
# plt.grid()
# plt.show()

# l = valid.select(pl.len()).collect().item()
# each = int(l/20)
# for i in valid.iter_slices(each):
# 	df = valid.slice(i*each, each).cast({v:pl.Float32 for v in (in_vars + out_vars)})
# 	print(df.slice(0, 1).collect())
# 	df.collect()
# 	df.write_parquet("Dataset/train/v3/train_{i}.parquet")