from train import train, valid
import utils, utils.dset as dset
import torch.utils.data as tdata
from torchmetrics.regression import R2Score
import torch, dill
import tqdm, time

iters_per_batch = 3

def train_loop(model, optimizer, loss, batch_size, norm_method, model_name, epochs=1, losses=[], train_losses=[], **kwargs):
	model.to(utils.nn.device)
	trs = dset.get_splits(norm_method, dset_class=dset.SQLDataset, fraction=0.01)
	valid_in, valid_out = next(iter(tdata.DataLoader(trs[-1],
			batch_sampler=tdata.BatchSampler(tdata.SequentialSampler(trs[-1]), batch_size=2_000, drop_last=False), #
			collate_fn=utils.nn.identity)))
	# valid_in, valid_out = next(iter(tdata.DataLoader(trs[-1], batch_size=1+int(batch_size//2), collate_fn=utils.nn.identity)))
	for j in tqdm.trange(2, 10, position=2, desc="Batches", ncols=100):
		batch = trs[j]
		pbar = tqdm.trange(epochs, position=1, desc="Epochs", ncols=100)
		vloss = valid(model, valid_in, valid_out)
		pbar.set_postfix_str(f"{vloss}")
		for i in pbar:
			# pbar = tqdm.tqdm(trs[:-4], position=0, desc="Batches")
			dataloader = tdata.DataLoader(batch, num_workers=4, prefetch_factor=2,
					batch_sampler=tdata.BatchSampler(tdata.RandomSampler(batch, generator=torch.Generator().manual_seed(i)), batch_size=batch_size, drop_last=False),
					collate_fn=utils.nn.identity)
			# dataloader = tdata.DataLoader(batch, num_workers=4, batch_size=batch_size, drop_last=False, collate_fn=utils.nn.identity,
			# 	prefetch_factor=2, shuffle=True)
			train_losses.append(train(model, dataloader, optimizer, loss))
			vloss = valid(model, valid_in, valid_out)
			pbar.set_postfix_str(f"{vloss}")
			losses.append(vloss)
			dill.dump({'model':model,
				'model_name':model_name,
				'optimizer':type(optimizer).__name__,
				'batch_size':batch_size,
				'norm_method':norm_method,
				'loss':type(loss).__name__,
				'losses':losses,
				'train_losses': train_losses,
				'time': time.strftime("%Y-%m-%d %H:%M:%S"),
				}, open(f"Models/{model_name}/model_checkpoint_batch_{j}_epoch_{i}.pickle", "wb"), byref=False, recurse=True)

def load_model(path):
	d = dill.load(open(path, 'rb'), ignore=True)
	if not isinstance(d, dict):
		return d
	if 'train losses' in d.keys():
		d['train_losses'] = d.pop('train losses')
	if d['loss'] == 'R2Score':
		d['loss'] = R2Score(num_outputs=utils.data.out_len).to(utils.nn.device)
	else:
		d['loss'] = getattr(torch.nn, d['loss'])().to(utils.nn.device)
		d['optimizer'] = getattr(torch.optim, d['optimizer'])(d['model'].parameters())
	return d

if __name__ == '__main__':
	print('enter main')
	torch.set_default_dtype(torch.float64)
	import CNN
	# model = fastkan.FasterKAN([utils.data.in_len, utils.data.out_len], grid_min=-100, grid_max=100, num_grids=12)
	# model = torch.load('kann.pickle')
	model = CNN.CNN()
	optimizer= torch.optim.RAdam(model.parameters(), lr=0.001, weight_decay=0.01)
	loss = torch.nn.MSELoss()
	batch_size = 20*384
	model_name = 'cnn' #'mlp_simple_bottleneck32batch_full_shuffle'
	norm = 'standardisation'
	epochs=5
	# restart_from_best = False

	import os
	if not os.path.exists(f"Models/{model_name}"):
		os.mkdir(f"Models/{model_name}")
		train_loop(model, optimizer, loss, batch_size, norm, model_name, epochs=epochs)
	else:
		import glob
		fname = f'Models/{model_name}/*.pickle'
		fname = glob.glob(fname)
		if len(fname) > 0:
			fname = fname[-1]
			print(fname)
			d = load_model(fname)
			optimizer = torch.optim.RAdam(d['model'].parameters())
			# d = {'model':d,
			# 	'model_name':model_name,
			# 	'optimizer':optimizer,
			# 	'batch_size':batch_size,
			# 	'norm_method':norm,
			# 	'loss':loss,
			# 	'losses':[],
			# 	'train_losses': [],
			# 	'time': time.strftime("%Y-%m-%d %H:%M:%S"),
			# 	}
			print(d['optimizer'], d['loss'], 'batch size ' + str(d['batch_size']), d['norm_method'], sep='|')
			print(d['model'])
			train_loop(epochs=epochs, **d)
		else:
			print(optimizer, loss, 'batch size ' + str(batch_size), norm, sep='|')
			print(model)
			train_loop(model, optimizer, loss, batch_size, norm, model_name, epochs=epochs)