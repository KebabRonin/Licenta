import polars as pl, numpy as np, os
from tqdm import tqdm
import utils, pickle, torch
from catboost import CatBoostRegressor
from torchmetrics.regression import R2Score

# cat_params = {
# 	'iterations': 10_000,
# 	'depth': 7,
# 	'task_type' : "GPU",
# 	'use_best_model': True,
# 	'eval_metric': 'R2', # R2Score(num_outputs=368).to('cuda'),
# 	'early_stopping_rounds': 300,
# 	# 'learning_rate': 0.01,
# 	'border_count': 32,
# 	'l2_leaf_reg': 3,
# 	# 'logging_level':"Verbose",
# 	'metric_period':500,
# }

# for var_name in tqdm(utils.data.out_vars):
# 	# if f"{var_name}_model.cbm" in os.listdir("CatBoostModel"):
# 	# 	continue
# 	print(var_name)
# 	model = CatBoostRegressor(**cat_params)
# 	model.fit(train_in, train_out.select(pl.col(var_name)).to_numpy(), eval_set=(valid_in, valid_out.select(pl.col(var_name)).to_numpy()))
# 	model.save_model(f"CatBoostModel/{var_name}_model.cbm")
# 	print(f"saved {var_name}")

class CatBoost(torch.nn.Module):
	def __init__(self):
		super().__init__()
		self.models = [pickle.load(open(f"../../impl/CatBoostModel/{var_name}_model.cbm", "rb")) for var_name in utils.data.out_vars]
	def forward(self, x):
		return np.concatenate([self.models[i].predict(x) for i in range(utils.data.out_len)], axis=0)
# async def train_batch(xs:torch.Tensor, ys:torch.Tensor, iters=1):
# 	for _ in range(iters):
# 		optimizer.zero_grad(set_to_none=True) # changed from False, supposedly better
# 		pred = model(xs)
# 		loss = loss_function(pred, ys)
# 		loss.backward()
# 		optimizer.step()
# 		# print(f"Loss:", loss.item())
# 	return loss.item()