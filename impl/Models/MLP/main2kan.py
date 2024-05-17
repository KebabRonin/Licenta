import polars as pl, numpy as np, torch
import sys, time, json, os, glob, threading, gc
# import asyncio
from my_utils import *
from tqdm import trange
from torchmetrics.regression import R2Score
import cloudpickle
sys.stdout.reconfigure(encoding='utf-8')

import matplotlib.pyplot as plt

import fastkan

torch.set_default_dtype(torch.float64)
DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
torch.set_default_device(DEVICE)

model = fastkan.FasterKAN([556, 556*2+1, 512, 368], num_grids=10)
model.to(DEVICE)

# import kan
# kan.create_dataset()
# kan.KAN([100, 199, 100])

train_files = [f"Dataset/train/v1/train_{i}.parquet" for i in range(49)] # Fara 49, 50, ca e de validare

# train_data = pl.scan_parquet("Dataset/train/v1/train_*.parquet").drop('sample_id')
# total_data_len = train_data.select(pl.len()).collect().item()

data_insights = json.load(open('data_insights.json'))


# === PARAMS ===
nr_batches = 10
batch_size = 20_000
iters_per_batch = 10
# none, +mean or +mean/std
normalization = "+mean"

# loss_function = R2Score(num_outputs=368).to(DEVICE)
loss_function = torch.nn.L1Loss().to(DEVICE)
# # optimizer = torch.optim.Adadelta(model.parameters(), lr = 1e-3, maximize=True)
optimizer = torch.optim.RAdam(model.parameters(), lr=1e-4) #, maximize=True)
# optimizer = torch.optim.SGD(model.parameters(), lr=1e-3, momentum=0.8, nesterov=True)

print_box(	f'Loss: {type(loss_function).__name__}',
			f'Optimizer: {type(optimizer).__name__}',
			f'Device: {DEVICE}',
			f'Nr. Batches: {nr_batches}',
			f'Batch size: {batch_size}',
			f'Iters per batch: {iters_per_batch}',
			f'Normalization: {normalization}',
			f'Start time: {time.strftime("%d-%b-%Y | %H:%M:%S")}')
_pbar_display_ncols = 100

print(model)


init_err, finl_err = [], []

def train_batch(xs:torch.Tensor, ys:torch.Tensor, iters=1, name="train"):
	global init_err, finl_err

	pbar = trange(iters, ncols=_pbar_display_ncols, position=1, leave=True, miniters=1, ascii=" 123456789@", desc=f"<{time.strftime('%H:%M')}> [{name[0]:02}] ({name[1]:2}) {name[2]:>6}")
	is_first = True
	for _ in pbar:
		optimizer.zero_grad(set_to_none=True)
		pred = model(xs)
		loss = loss_function(pred, ys)
		loss.backward()
		optimizer.step()
		pbar.set_postfix_str(f"{loss.item():.10f}")
		if is_first:
			is_first = False
			init_err.append(loss.item())
	del xs, ys
	finl_err.append(loss.item())
	torch.save(model, f"model_checkpoint_{time.strftime('%d-%b-%Y-%H-%M-%S')}.pt", pickle_module=cloudpickle)

def read_data(data: pl.LazyFrame, offset: int, batch_size):
	# t0 = time.time()
	batch = data.slice(offset=offset, length=batch_size).collect()

	train_in = normalize_subset(batch,  in_vars, method=normalization).to_numpy()
	train_out = normalize_subset(batch,  out_vars, method=normalization).to_numpy()

	# train_in  = torch.tensor(train_in, device=DEVICE)
	# train_out = torch.tensor(train_out, device=DEVICE)

	# t1 = time.time()
	# print(f"Read start ({offset}): {time.strftime('%H:%M:%S', time.localtime(t0))}")
	# print(f"Read end   ({offset}): {time.strftime('%H:%M:%S', time.localtime(t1))}")
	# print(f"Read     : {int(t1-t0)//60:02}:{(t1-t0)%60:05.2f}s")
	return train_in, train_out

def main():
	th = None
	time.sleep(1)
	pbar = trange(nr_batches, ncols=_pbar_display_ncols, position=0, leave=True, desc="Batches", miniters=1)
	for iter_nr in pbar:
		file_nr = np.random.randint(0, 40)
		data = pl.scan_parquet(f"Dataset/train/v1/train_{file_nr}.parquet").drop("sample_id")
		train_len = data.select(pl.len()).collect().item()
		offset = np.random.randint(0, train_len - batch_size)

		train_in, train_out = read_data(data, offset, batch_size)

		if th:
			plott(init_err, finl_err)
			gc.collect()
			# print("Waiting for GPU model train to finish")
			th.join()

		train_in  = torch.tensor(train_in, device=DEVICE)
		train_out = torch.tensor(train_out, device=DEVICE)

		th = threading.Thread(target=train_batch, args=(train_in, train_out, iters_per_batch, (iter_nr, file_nr, offset)))
		th.start()
	else:
		# gc.collect()
		th.join()

	torch.save(model, "model.pt", pickle_module=cloudpickle)
	# torch.save(optimizer.state_dict(), "optim.pt", pickle_module=cloudpickle)

	# Cleanup intermediate models
	for f in (glob.glob(r"model_checkpoint*.pt") + glob.glob(r"optim_checkpoint*.pt")):
		try:
			os.remove(f)
		except FileNotFoundError:
			pass

	print(f"\n**{'Finished':=^42}**\n\n")
	plott(init_err, finl_err)
	return


if __name__ == '__main__':
	main()

# === Rejects ===


# @torch.jit.script
# model = torch.compile(model)

# loss_function = torch.nn.L1Loss()

# from torch.utils.data import Dataset, DataLoader
# train_data = MyDataset(train_files)
# train_dataloader = DataLoader(train_data, batch_size=batch_size, shuffle=True)

# train_data = pyarrow.parquet.ParquetDataset(train_files)

# def score_model(model):
# 	model.eval()
# 	l1 = loss_function()
# 	print(l1)
# 	model.train()
