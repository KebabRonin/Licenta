import polars as pl, numpy as np, torch
# from torch.utils.data import Dataset, DataLoader
import sys, time, gc, json, os
from my_utils import in_vars, out_vars
from tqdm import trange
from torchmetrics.regression import R2Score
sys.stdout.reconfigure(encoding='utf-8')

# from torch.profiler import profile, record_function, ProfilerActivity
torch.set_default_dtype(torch.float64)

from model_def import model
try:
	model = torch.load('model.pt').double()
except Exception as e:
	print("Didn't load model, training from scratch... (Error is:", e)

train_files = [f"Dataset/train/v1/train_{i}.parquet" for i in range(49)] # Fara 49, 50, ca e de validare

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print("Device:", device)
train_data = pl.scan_parquet("Dataset/train/v1/train_*.parquet").drop('sample_id')
# train_data = pyarrow.parquet.ParquetDataset(train_files)
data_insights = json.load(open('data_insights.json'))
# valid_data =
# def score_model(model):
# 	model.eval()
# 	l1 = loss_function()
# 	print(l1)
# 	model.train()



batch_size = 55_000

model.to(device)

# loss_function = torch.nn.L1Loss()
loss_function = R2Score(num_outputs=368).to('cuda')
optimizer = torch.optim.Adam(model.parameters(), amsgrad=True, maximize=True)
try:
	optimizer.load_state_dict(torch.load('optim.pt'))
	optimizer.param_groups[0]['fused']=False
	print(optimizer.state_dict()['param_groups'])
except Exception as e:
	print("Didn't load optimizer, training from scratch... (Error is:", e)

def train_batch(xs:torch.Tensor, ys:torch.Tensor, iters=1):
	for _ in range(iters):
		optimizer.zero_grad(set_to_none=True) # changed from False, supposedly better
		pred = model(xs)
		loss = loss_function(pred, ys)
		loss.backward()
		optimizer.step()
		# print(f"Loss:", loss.item())
	return loss.item()
	# # m = torch.jit.script(model)


train_len = train_data.select(pl.len()).collect().item()

# train_data = MyDataset(train_files)
# train_dataloader = DataLoader(train_data, batch_size=batch_size, shuffle=True)

for j in trange(5, bar_format="{l_bar}{bar}{r_bar:<32}\n"):
	offset = np.random.randint(0, train_len - batch_size)
	t0 = time.time()
	batch = train_data.slice(offset=0, length=batch_size).collect()
	train_in = torch.tensor(batch.select(
		(pl.col(in_vars[i]) -
		data_insights[in_vars[i]]['mean']) #/ (data_insights[in_vars[i]]['std_dev'] if data_insights[in_vars[i]]['std_dev'] != 0 else 1)
		for i in range(len(in_vars))
	).to_numpy(), device=device)
	train_out = torch.tensor(batch.select(
		(pl.col(out_vars[i]) -
		data_insights[out_vars[i]]['mean']) #/ (data_insights[out_vars[i]]['std_dev'] if data_insights[out_vars[i]]['std_dev'] != 0 else 1)
		for i in range(len(out_vars))
	).to_numpy(), device=device)
	t1 = time.time()
	loss = train_batch(train_in, train_out, iters=30)
	print(f"Off {offset} | Read {t1 - t0:>20} | Train {time.time() - t1:>20} | Loss {loss}")
	torch.save(model, f"model_checkpoint{time.time()}.pt")
	torch.save(optimizer.state_dict(), f"optim_checkpoint{time.time()}.pt")

torch.save(optimizer.state_dict(), "optim.pt")
torch.save(model, "model.pt")