import polars as pl, numpy as np, torch
import sys, time, json, os, glob, threading, gc
# import asyncio
from my_utils import *
from tqdm import trange, tqdm
from torchmetrics.regression import R2Score
import cloudpickle
sys.stdout.reconfigure(encoding='utf-8')

import matplotlib.pyplot as plt

# from torch.profiler import profile, record_function, ProfilerActivity

DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
# torch.set_default_device(DEVICE)
torch.set_default_dtype(torch.float64)

def load_model():
	try:
		model = torch.load('model.pt')
		print("Loaded model weights")
	except Exception as e:
		print("Didn't load model, training from scratch... (Error is:", e)
		from model_def_miine import model

	return model

# train_files = [f"Dataset/train/v1/train_{i}.parquet" for i in range(49)] # Fara 49, 50, ca e de validare

data_insights = json.load(open('data_insights.json'))
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader
# === PARAMS ===
NUM_EPOCHS = 1
batch_size = 256
iters_per_batch = 1
# none, +mean or +mean/std
normalization = "+mean/std"
CHECKPOINT_INTERVAL = 15

_pbar_display_ncols = 100

def main():
	# loss_function = R2Score(num_outputs=368).to(DEVICE)
	model = load_model().to(DEVICE)
	loss_function = torch.nn.MSELoss().to(DEVICE)
	# optimizer = torch.optim.Adadelta(model.parameters(), lr = 1e-3, maximize=True)
	optimizer = torch.optim.RAdam(model.parameters(), lr=1e-3, ) #, maximize=True)
	# optimizer = torch.optim.Adadelta()
	# optimizer = torch.optim.SGD(model.parameters(), lr=1e-3, momentum=0.8, nesterov=True)
	print_box(	f'Loss: {type(loss_function).__name__}',
				f'Optimizer: {type(optimizer).__name__}',
				f'Device: {DEVICE}',
				f'Nr. Epochs: {NUM_EPOCHS}',
				f'Batch size: {batch_size}',
				f'Iters per batch: {iters_per_batch}',
				f'Normalization: {normalization}',
				f'Start time: {time.strftime("%d-%b-%Y | %H:%M:%S")}')

	print(model)
	time.sleep(1)
	dset = CustomSQLDataset()
	tr, val = tdata.random_split(dset, [0.9, 0.1], generator=torch.Generator().manual_seed(50))
	sqldloader = DataLoader(tr, num_workers=5,
							batch_sampler=tdata.BatchSampler(tdata.RandomSampler(tr, generator=torch.Generator().manual_seed(50)), batch_size=batch_size, drop_last=False), #
							collate_fn=identity,
							prefetch_factor=5)
	pbar = tqdm(sqldloader, ncols=_pbar_display_ncols, position=0, leave=True, desc="Batches", miniters=1)
	for e in trange(NUM_EPOCHS, position=2, leave=True, desc=f"Epochs", miniters=1):
		i = 0
		for train_in, train_out in pbar:
			i += 1
			# print(train_in.device, train_out.device)
			train_in, train_out = train_in.to(DEVICE), train_out.to(DEVICE)
			optimizer.zero_grad(set_to_none=True)
			pred = model(train_in)
			loss = loss_function(pred, train_out)
			loss.backward()
			optimizer.step()
			pbar.set_postfix_str(f"{loss.item():.10f}", refresh=False)
			if i % CHECKPOINT_INTERVAL == CHECKPOINT_INTERVAL - 1:
				torch.save(model, f"model_checkpoint_{time.strftime('%d-%b-%Y-%H-%M-%S')}.pt", pickle_module=cloudpickle)
	torch.save(model, "model.pt", pickle_module=cloudpickle)
	for f in (glob.glob(r"model_checkpoint*.pt") + glob.glob(r"optim_checkpoint*.pt")):
		try:
			os.remove(f)
		except FileNotFoundError:
			pass


	print(f"\n**{'Finished':=^42}**\n")
	return model


if __name__ == '__main__':
	model = main()
	from test import validate_model
	validate_model(model)

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
