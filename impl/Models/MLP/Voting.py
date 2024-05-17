import numpy as np, torch, math
import polars as pl, sys
from my_utils import *
sys.stdout.reconfigure(encoding='utf-8')
batch_size = 5


train_data = pl.scan_parquet("Dataset/train/v1/train_*.parquet", n_rows=batch_size).drop("sample_id").collect()
train_in  = train_data.select(pl.col(in_vars ))
train_out = train_data.select(pl.col(out_vars))
in_len=556
out_len=368

class EnsambleModel:
	def __init__(self, name):
		self.name = name
	def predict(self, data:np.ndarray):
		# print(data.shape)
		return np.zeros((data.shape[0], out_len))

class TorchModel(EnsambleModel):
	def __init__(self, name, model_file):
		super().__init__(name)
		self.model_file = model_file
		self.model = torch.load(model_file)
		self.model.eval()
		self.reset()
	def predict(self, data:np.ndarray):
		self.reset()
		self.last_rez = self.model(torch.tensor(data, device='cuda'))
			# print(self.last_data)
		return self.last_rez.cpu().numpy().transpose().squeeze()
	def reset(self):
		del self.last_rez
		self.last_rez = None

data = None
models = [EnsambleModel("average"), EnsambleModel("average2")]
preds = np.stack([model.predict(data) for model in models], axis=-1)
n_models = len(models)
# ReLU + Softmax
class Voter(torch.nn.Module):
	def __init__(self):
		super().__init__()
		self.voters = torch.nn.ModuleList([torch.nn.Sequential(
			torch.nn.Linear(n_models, 1),
			torch.nn.LeakyReLU()
		) for _ in range(in_len)])
	def forward(self, x):
		return torch.concat([self.voters[i](x[:, i, :].squeeze()) for i in range(in_len)])
# with torch.no_grad():
# 	EM_en = EMRegressor([EnsambleModel("average"), EnsambleModel("average2")]) #, TorchModel("MLP", "model.pt")])

# 	EM_en.train(train_in.to_numpy(), train_out.to_numpy(), n_iters=1)