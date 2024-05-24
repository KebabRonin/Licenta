import numpy as np, torch, math, time
import polars as pl, sys
from my_utils import *
from tqdm import trange
from torchmetrics.regression import R2Score
sys.stdout.reconfigure(encoding='utf-8')


DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
in_len=556
out_len=368

# (self.predictions[None, :, :] * self.weights)
def normalize_tensor(tensor: torch.Tensor, method='none', denormalize=True):
	if tensor.shape[0] == 556:
		means = in_means
		stds = in_std_dev
	elif tensor.shape[0] == 368:
		means = out_means
		stds = out_std_dev
	elif tensor.shape[0] == (556 + 368):
		means = np.concatenate(in_means, out_means)
		stds = np.concatenate(in_std_dev, out_std_dev)

	stds[stds == 0] = 1
	means = torch.tensor(means, device=tensor.device)
	stds  = torch.tensor(stds , device=tensor.device)

	match method:
		case "+mean/std":
			if denormalize:
				return (tensor[None, :] * stds)[None, :] + means
			else:
				return (tensor[None, :] - means)[None, :] / stds
		case "+mean":
			if denormalize:
				return tensor[None, :] + means
			else:
				return tensor[None, :] - means
		case "none":
			return tensor
		case _:
			raise Exception("'method' not recognized. Must be callable or one of ['+mean/std', '+mean', 'none']")


class EnsambleModel:
	def __init__(self, name, normalization):
		self.name = name
		self.normalization = normalization
	def predict(self, data_in:np.ndarray):
		data_in = normalize_subset(data_in, method=self.normalization)
		data_in  = torch.tensor(data_in, device=DEVICE)
		data_out = self._predict(data_in)
		return normalize_subset(data_out, method=self.normalization, denormalize=True)

class ConstantModel(EnsambleModel):
	def __init__(self, name, val=0):
		super().__init__(name, normalization="none")
		self.name = name
		self.val = val
	def _predict(self, data:np.ndarray):
		return np.full((data.shape[0], out_len), fill_value=self.val)

class TorchModel(EnsambleModel):
	def __init__(self, name, model_file, normalization="+mean/std"):
		super().__init__(name, normalization=normalization)
		self.model_file = model_file
		self.model = torch.load(model_file)
		self.model.eval()
	def _predict(self, data:np.ndarray):
		return self.model(torch.tensor(data, device='cuda'))

class Voter(torch.nn.Module):
	def __init__(self, n_models):
		super().__init__()
		self.n_models = n_models
		# Separate weights for each out variable
		self.voters = torch.nn.ModuleList([torch.nn.Linear(self.n_models, 1) for _ in range(out_len)])
	def forward(self, x):
		# x dims: n_samples x vars x voters
		return torch.concat([self.voters[i](x[:, i, :].squeeze()) for i in range(out_len)])


def predict_models(models, data_in):
	predictions = np.stack([model.predict(data_in) for model in models], axis=-1)
	print(f"{predictions.shape=}")
	return predictions

def read_data(data: pl.LazyFrame, offset: int, batch_size, normalization: str):

	data = data.slice(offset=offset, length=batch_size).collect()

	train_in = normalize_subset(data,  in_vars, method="none")
	train_out = normalize_subset(data,  out_vars, method="none")
	return train_in, train_out

models = [EnsambleModel("average", 0), EnsambleModel("average2", 1)] #, TorchModel("MLP", "model.pt")]
n_models = len(models)
voter = Voter(n_models)
batch_size = 10
nr_batches = 10
loss_function = R2Score(num_outputs=368).to(DEVICE)
optimizer = torch.optim.AdamW(voter.parameters(), maximize=True)

print_box(	f'Loss: {type(loss_function).__name__}',
			f'Optimizer: {type(optimizer).__name__}',
			f'Device: {DEVICE}',
			f'Nr. Batches: {nr_batches}',
			f'Batch size: {batch_size}',
			f'Start time: {time.strftime("%d-%b-%Y | %H:%M:%S")}')

file_nr = np.random.randint(0, 40)
data = pl.scan_parquet(f"Dataset/train/v1/train_{file_nr}.parquet").drop("sample_id")
train_len = data.select(pl.len()).collect().item()
offset = np.random.randint(0, train_len - batch_size)

train_in, train_out = read_data(data, offset, batch_size, normalization='none')
preds = predict_models(models, train_in)
print(preds)
exit(0)
for _ in trange(nr_batches, miniters=1):
	file_nr = np.random.randint(0, 40)
	data = pl.scan_parquet(f"Dataset/train/v1/train_{file_nr}.parquet").drop("sample_id")
	train_len = data.select(pl.len()).collect().item()
	offset = np.random.randint(0, train_len - batch_size)

	train_in, train_out = read_data(data, offset, batch_size)
	preds = predict_models(models, train_in)

	final_pred = voter(preds)
	loss = loss_function(final_pred, preds)
	loss.backward()
	print(loss.item())
	optimizer.step()


# 	EM_en.train(train_in.to_numpy(), train_out.to_numpy(), n_iters=1)