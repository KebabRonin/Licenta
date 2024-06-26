import polars as pl, numpy as np, os
from my_utils import *
from catboost import CatBoostRegressor
from tqdm import tqdm

# === READ DATA ===

# train_files = [f"../impl/Dataset/train/v1/train_{i}.parquet" for i in range(49)] # Fara 49, 50, ca e de validare

train_data = pl.scan_parquet("../impl/Dataset/train/v1/train_*.parquet").drop('sample_id')
train_len = train_data.select(pl.len()).collect().item()

# === PARAMS ===
nr_batches = 30
iters_per_batch = 20
batch_size = 500_000
valid_size = 50_000

batch = train_data.slice(offset=0, length=batch_size).collect()
train_in = normalize_subset(batch,in_vars).to_numpy()
train_out = normalize_subset(batch,out_vars)
valid_data = pl.scan_parquet("../impl/Dataset/train/v1/train_49.parquet", n_rows=valid_size).drop('sample_id').collect()
valid_in = normalize_subset(valid_data, in_vars).to_numpy()
valid_out = normalize_subset(valid_data, out_vars)

cat_params = {
	'iterations': 10_000,
	'depth': 8,
	'task_type' : "GPU",
	'use_best_model': True,
	'eval_metric': 'R2', # R2Score(num_outputs=368).to('cuda'),
	'early_stopping_rounds': 300,
	# 'learning_rate': 0.01,
	'border_count': 32,
	'l2_leaf_reg': 3,
	# 'logging_level':"Verbose",
	'metric_period':500,
}

for var_name in tqdm(['ptend_q0002_17']):
	# if var_name in const_out_vars or f"{var_name}_model.cbm" in os.listdir("CatBoostModel8"):
	# 	continue
	print(var_name)
	model = CatBoostRegressor(**cat_params)
	model.fit(train_in, train_out.select(pl.col(var_name)).to_numpy(), eval_set=(valid_in, valid_out.select(pl.col(var_name)).to_numpy()))
	model.save_model(f"CatBoostModel8/{var_name}_model.cbm")
	print(f"saved {var_name}")


# async def train_batch(xs:torch.Tensor, ys:torch.Tensor, iters=1):
# 	for _ in range(iters):
# 		optimizer.zero_grad(set_to_none=True) # changed from False, supposedly better
# 		pred = model(xs)
# 		loss = loss_function(pred, ys)
# 		loss.backward()
# 		optimizer.step()
# 		# print(f"Loss:", loss.item())
# 	return loss.item()