import polars as pl, numpy as np, torch
import sys, time, json, os, glob, threading, gc
# import asyncio
from my_utils import *
from my_test import validate_model
from tqdm import trange, tqdm
from torchmetrics.regression import R2Score
import cloudpickle
sys.stdout.reconfigure(encoding='utf-8')

import matplotlib.pyplot as plt

# from torch.profiler import profile, record_function, ProfilerActivity

DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
# torch.set_default_device(DEVICE)
torch.set_default_dtype(torch.float64)

# train_files = [f"Dataset/train/v1/train_{i}.parquet" for i in range(49)] # Fara 49, 50, ca e de validare

data_insights = json.load(open('data_insights.json'))
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader
# === PARAMS ===
NUM_EPOCHS = 1
batch_size = 4096
ITERS_PER_BATCH = 100
# none, +mean or +mean/std
NORMALIZATION = "+mean/std"

_pbar_display_ncols = 100

def load_model():
	tbatches = 0
	trail = []
	normalization = NORMALIZATION
	try:
		model = torch.load('model.pt')
		if isinstance(model, dict):
			normalization = model['normalization']
			tbatches = model['total_batches']
			if 'trail' in model:
				trail = model['trail']
			model = model['model']
		print("Loaded model weights")
	except Exception as e:
		print("Didn't load model, training from scratch... (Error is:", e)
		# from model_def import model
		from model_def_kan import model
		# from fastkan import FasterKAN
		# model = FasterKAN([556, 556*2, 556*2, 556*2, 368])

	return model, tbatches, normalization, trail

def main():
	model, tbatches, normalization, trail = load_model()
	model.to(DEVICE)
	print(f"{tbatches=}")
	loss_function = torch.nn.MSELoss().to(DEVICE)
	# loss_function = R2Score(num_outputs=368).to(DEVICE)
	# optimizer = torch.optim.Adadelta(model.parameters(), lr = 1e-3, maximize=True)
	# optimizer = torch.optim.Adadelta(model.parameters(), lr=0.014666056303807039, ) #, maximize=True)
	optimizer = torch.optim.RAdam(model.parameters(),lr=1e-6)
	# optimizer = torch.optim.LBFGS(model.parameters(), history_size=10)
	# optimizer = torch.optim.SGD(model.parameters(), lr=1e-3, momentum=0.8, nesterov=True)
	print_box(	f'Loss: {type(loss_function).__name__}',
				f'Optimizer: {type(optimizer).__name__}',
				f'Device: {DEVICE}',
				f'Nr. Epochs: {NUM_EPOCHS}',
				f'Batch size: {batch_size}',
				f'Iters per batch: {ITERS_PER_BATCH}',
				f'Normalization: {normalization}',
				f'Start time: {time.strftime("%d-%b-%Y | %H:%M:%S")}')

	print(model)
	time.sleep(1)
	dset = CustomSQLDataset(norm_method=normalization)
	splits = get_splits()
	# print(splits)
	trs = tdata.random_split(dset, splits, generator=torch.Generator().manual_seed(50))
	# for batch in tqdm(trs[:1], position=0, leave=True, desc=f"Batches", miniters=1, ncols=_pbar_display_ncols): #tbatches%(len(splits)-VALID_RESERVE):-VALID_RESERVE
	batch = trs[0]
	for ib in trange(ITERS_PER_BATCH, position=1, leave=True, desc=f"Iterations this batch", miniters=1, ncols=_pbar_display_ncols):
		sqldloader = DataLoader(batch, num_workers=4,
								batch_sampler=tdata.BatchSampler(tdata.RandomSampler(batch, generator=torch.Generator().manual_seed(ib)), batch_size=batch_size, drop_last=False), #
								collate_fn=identity,
								prefetch_factor=1)
		pbar = tqdm(sqldloader, ncols=_pbar_display_ncols, position=2, leave=True, desc="Mini-Batches", miniters=1)
		for train_in, train_out in pbar:
			# print(train_in.device, train_out.device)
			train_in, train_out = train_in.to(DEVICE), train_out.to(DEVICE)
			optimizer.zero_grad(set_to_none=True)
			pred = model(train_in)
			loss = loss_function(pred, train_out)
			loss.backward()
			optimizer.step()
			pbar.set_postfix_str(f"{loss.item():.10f}", refresh=False)
		tbatches += 1
		trail.append(validate_model(model))
		print("Val R2:", trail[-1])
		torch.save({
			'model':model,
			'trail': trail,
			'total_batches':tbatches,
			'normalization':normalization,
			'loss':type(loss_function).__name__,
			'optimizer': type(optimizer).__name__,
			}, f"model_checkpoint_{time.strftime('%d-%b-%Y-%H-%M-%S')}.pt", pickle_module=cloudpickle)
		print(f"Saved model checkpoint {tbatches=}")
	torch.save({
		'model':model,
		'trail': trail,
		'total_batches':tbatches,
		'normalization':normalization,
		'loss':type(loss_function).__name__,
		'optimizer': type(optimizer).__name__,
		}, "model.pt", pickle_module=cloudpickle)
	print("Saved model")
	for f in (glob.glob(r"model_checkpoint*.pt") + glob.glob(r"optim_checkpoint*.pt")):
		try:
			os.remove(f)
		except FileNotFoundError:
			pass


	print(f"\n**{'Finished':=^42}**\n")
	return model


if __name__ == '__main__':
	model = main()
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
