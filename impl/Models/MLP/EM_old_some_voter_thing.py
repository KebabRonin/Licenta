import numpy as np, torch, math
import polars as pl, sys
from my_utils import *
from torchmetrics.regression import R2Score
from torch.nn import L1Loss
sys.stdout.reconfigure(encoding='utf-8')
batch_size = 100


train_data = pl.scan_parquet("train_ex.parquet").drop("sample_id").collect()#("Dataset/train/v1/train_*.parquet", n_rows=batch_size).drop("sample_id").collect()
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

class ConstantModel(EnsambleModel):
	def __init__(self, name, value):
		super().__init__(name)
		self.value = value
	def predict(self, data:np.ndarray):
		# print(data.shape)
		return np.full((data.shape[0], out_len), fill_value=self.value, dtype=np.float64)

class TorchModel(EnsambleModel):
	def __init__(self, name, model_file):
		super().__init__(name)
		self.model_file = model_file
		self.model = torch.load(model_file)
		self.model.eval()
		self.last_rez = None
	def predict(self, data:np.ndarray):
		self.reset()
		self.last_rez = self.model(torch.tensor(data, device='cuda'))
			# print(self.last_data)
		return self.last_rez.cpu().numpy().squeeze()
	def reset(self):
		del self.last_rez
		self.last_rez = None

# ReLU + Softmax
class EMRegressor:
	def __init__(self, models:list[EnsambleModel], in_len=556, out_len=368):
		self.models = models
		self.model_names = [m.name for m in self.models]
		self.nr_models = len(self.models)
		self.in_len = in_len
		self.out_len = out_len
		self.predictions = None
		self.weights = np.full(shape=(out_len, self.nr_models), fill_value=1/self.nr_models, dtype=np.float32)

	def predict_models(self, data: np.ndarray):
		rezs = [model.predict(data) for model in self.models] # ca sa nu fie problema cu predict 0 (??)
		print(f"{rezs=}")
		for i in rezs:
			print(i.shape)
		self.predictions = np.stack(rezs, axis=-1)

	def predict(self, data=None):
		if self.predictions is None:
			if data is None:
				raise Exception("predict() can only be called after predict_models() or with the 'data' argument set")
			else:
				self.predict_models(data)

		print(f"{self.weights.shape=}, {self.predictions.shape=}")
		self.final_pred = (self.predictions[None, :, :] * self.weights).squeeze(axis=0).sum(axis=2).squeeze()
		print(f"{self.final_pred.shape=}")
		print(f"{self.final_pred=}")
		return self.final_pred

	def Estep(self, dout):
		mulw = (self.predictions[None, :, :] * self.weights).squeeze(axis=0)
		other = mulw[:, :, None].squeeze(axis=2) # prepare for elementwise division
		fp = self.final_pred[:, :, np.newaxis]   # prepare for elementwise division
		print(f"{mulw.shape=}, {other.shape=}, {fp.shape=}")
		self.z = other / fp # dout[:, :, np.newaxis]
		print(f"EStep {self.z.shape=}")
		print(f"{self.z=}")

	def Mstep(self):
		print(f"{self.weights[0]=}")
		self.weights = np.mean(self.z, axis=0)
		print(f"MStep {self.weights.shape=}")
		print(f"{self.weights[0]=}")

	def train(self, data_in: np.ndarray, data_out: np.ndarray, n_iters=10):
		if data_in.shape[1] != self.in_len:
			raise Exception(f"Input data shape ({data_in.shape}) doesn't fit with the specified input feature length ({self.in_len})")
		# self.predict(data_in) # updates self.prediction matrix (samples x output features x models)
		for i in range(n_iters):
			self.predict(data_in) # updates self.prediction matrix (samples x output features x models)
			self.Estep(data_out)
			self.Mstep()
			print(f"Iter {i+1:>3} weights:\n{self.weights[:20, :].T}")

with torch.no_grad():
	EM_en = EMRegressor([ConstantModel("average", 0), ConstantModel("average2", 1), TorchModel("MLP", "model.pt")])

	EM_en.train(train_in.to_numpy(), train_out.to_numpy(), n_iters=20)

	val_data = pl.scan_parquet("Dataset/train/v1/train_40.parquet", n_rows=batch_size).drop("sample_id").collect()
	v_in  = torch.tensor(train_data.select(pl.col(in_vars )).to_numpy())
	v_out = torch.tensor(train_data.select(pl.col(out_vars)).to_numpy())
	pred = EM_en.predict(v_in)
	r2score = R2Score(num_outputs=368, multioutput="raw_values")
	maescore = L1Loss()
	print("r2:", r2score(pred, v_out))
	for i in range(3):
		print(i, "r2", r2score(EM_en.predictions[:,:,i].squeeze(), v_out))
	print("mae:", maescore(torch.tensor(pred), v_out))

