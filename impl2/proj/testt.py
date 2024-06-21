import utils.data
import utils.dset as dset, utils, utils.nn
import seaborn as sb, numpy as np
import torch.utils.data as tdata
from torchmetrics.regression import R2Score
import torch, dill
import tqdm


def test(model, ins, outs):
	print('ad')
	ins, outs = ins.to(utils.nn.device), outs.to(utils.nn.device)
	print('ad')
	r2score = R2Score(num_outputs=utils.data.out_len, multioutput='raw_values').to(utils.nn.device)
	mse = torch.nn.MSELoss().to(utils.nn.device)
	model.eval()
	with torch.no_grad():
		prediction = model(ins)
		loss_values = r2score(prediction, outs).cpu()
		# losss = mse(prediction, outs).cpu()
		# header=f"{'Name':^15}|{'Actual':^10}|{'Pred':^10}|{'Diff':^10}|{'R2':^10}|*|"
		# print(f"{'&':=^186}")
		# print(*(header for _ in range(3)), sep='')
		# print(f"{'&':=^186}")
		# fstr = "{vname:<15}|{act:>10.5f}|{pred:>10.5f}|{diff:>10.5f}|{r2ll:>10.5f}|*|"
		# for i in range(0, utils.data.out_len, 3):
		# 	print(*(
		# 		fstr.format(
		# 			vname=utils.data.out_vars[idx],
		# 			act=outs[0][idx],
		# 			pred=prediction[0][idx],
		# 			diff=outs[0][idx] - prediction[0][idx],
		# 			r2ll=loss_values[idx],
		# 		) for idx in range(i, i+3) if idx < utils.data.out_len
		# 	), sep='')
		# print(f"{'&':=^186}")
		# print("MSE", losss.item())
	return loss_values

def load_model(path):
	d = dill.load(open(path, 'rb'), ignore=True)
	if 'train losses' in d.keys():
		d['train_losses'] = d.pop('train losses')
	d['optimizer'] = getattr(torch.optim, d['optimizer'])(d['model'].parameters())
	if d['loss'] == 'R2Score':
		d['loss'] = R2Score(num_outputs=utils.data.out_len).to(utils.nn.device)
	else:
		d['loss'] = getattr(torch.nn, d['loss'])().to(utils.nn.device)
	return d
def plot_correlation(test_in, test_out):
	xticks, yticks = utils.data.in_ticks, utils.data.out_ticks

	test_in, test_out = test_in.cpu().numpy(), test_out.cpu().numpy()

	corm = np.zeros((utils.data.out_len, utils.data.in_len))
	for i in tqdm.trange(utils.data.in_len, position=0, desc="Inputs", ncols=100):
		for j in range(utils.data.out_len):
			m = np.corrcoef(test_out[:, j], test_in[:, i])
			corm[j, i] = m[0, 1] # getcov(out, in)
	plt.matshow(corm)

	print(corm[20, 490])

	plt.xticks(*xticks, ha='left')
	plt.yticks(*yticks)
	plt.grid()
	plt.colorbar()
	plt.show()

def plot_distr(test_in, test_out):
	fig = plt.figure()
	ax = fig.add_subplot(projection='3d')

if __name__ == '__main__':
	torch.set_default_dtype(torch.float64)
	# d = load_model('Models/resnet_parallel/model_checkpoint_batch_4_epoch_4.pickle')
	batches=2
	import matplotlib.pyplot as plt
	#d['norm_method']
	trs = dset.get_splits('minmax100', dset_class=dset.TimestepSQLDataset, fraction=0.05)

	test_in, test_out = next(iter(tdata.DataLoader(trs[-1], batch_size=batches, drop_last=False, collate_fn=utils.nn.identity)))

	# plot_correlation(test_in, test_out)
	# plt.hist(test_in.cpu().numpy()[:, 0], bins=100)
	# plt.stackplot(range(556), test_in.cpu().numpy()[0, :])
	# plt.show()
	# exit()
	# model = torch.load('../impl/model.pt')
	model = load_model('Models/kan/model_checkpoint_batch_4_epoch_29.pickle')
	if isinstance(model, dict):
		d = model
		model = d['model'].to(utils.nn.device)

	errs = test(model, test_in, test_out)
	print(errs.mean())
	print(torch.tensor(list(filter(lambda x: x > 0, errs))).mean())

	plt.plot(errs, label=f'')
	plt.axis((0, 367, -0.5, 1.1))
	xticks = utils.data.out_ticks
	plt.xticks(*xticks, ha='left')
	plt.grid()
	plt.show()

	r2score = R2Score(num_outputs=utils.data.out_len).to(utils.nn.device)
	model.eval()
	with torch.no_grad():
		import numpy as np, xarray as xr
		from mpl_toolkits.basemap import Basemap
		file = r'ClimSim_low-res_grid-info.nc'
		grid = xr.open_dataset(file,engine='netcdf4')
		ins, outs = test_in.to('cpu').numpy(), test_out.to('cpu').numpy()
		losses = []
		for i in range(384):
			inn, outt = ins[[i + j*384 for j in range(batches)], :], outs[[i + j*384 for j in range(batches)], :]
			inn, outt = torch.tensor(inn, dtype=torch.float64).to(utils.nn.device), torch.tensor(outt, dtype=torch.float64).to(utils.nn.device)
			prediction = model(inn)
			losses.append(r2score(prediction, outt).cpu().numpy().mean())

		m = Basemap(projection='robin',lon_0=0,resolution='c')
		x,y = m(grid.lon,grid.lat)
		c = 'ptend_q0001_40'
		m.drawcoastlines(linewidth=0.5)
		plot = plt.tricontourf(x,y,losses,cmap='viridis',levels=14)
		m.colorbar(plot)
		plt.show()
	# train_loop(model, optimizer, loss, batch_size, 'minmax100')