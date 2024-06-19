import ResNet
from train import train, valid
import utils.data
import utils.dset as dset, utils
import torch.utils.data as tdata
from torchmetrics.regression import R2Score
import torch, dill
import tqdm


def test(model, ins, outs):
	ins, outs = ins.to(utils.nn.device), outs.to(utils.nn.device)
	r2score = R2Score(num_outputs=utils.data.out_len, multioutput='raw_values').to(utils.nn.device)
	mse = torch.nn.MSELoss().to(utils.nn.device)
	model.eval()
	with torch.no_grad():
		prediction = model(ins)
		loss_values = r2score(prediction, outs).cpu()
		losss = mse(prediction, outs).cpu()
		header=f"{'Name':^15}|{'Actual':^10}|{'Pred':^10}|{'Diff':^10}|{'R2':^10}|*|"
		print(f"{'&':=^186}")
		print(*(header for _ in range(3)), sep='')
		print(f"{'&':=^186}")
		fstr = "{vname:<15}|{act:>10.5f}|{pred:>10.5f}|{diff:>10.5f}|{r2ll:>10.5f}|*|"
		for i in range(0, utils.data.out_len, 3):
			print(*(
				fstr.format(
					vname=utils.data.out_vars[idx],
					act=outs[0][idx],
					pred=prediction[0][idx],
					diff=outs[0][idx] - prediction[0][idx],
					r2ll=loss_values[idx],
				) for idx in range(i, i+3) if idx < utils.data.out_len
			), sep='')
		print(f"{'&':=^186}")
		print("MSE", losss.item())
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

if __name__ == '__main__':
	torch.set_default_dtype(torch.float64)
	d = load_model(r'Models\kan\model_checkpoint_batch_0_epoch_3.pickle')
	batches=5
	import matplotlib.pyplot as plt

	trs = dset.get_splits(d['norm_method'], dset_class=dset.TimestepSQLDataset, fraction=0.05)

	test_in, test_out = next(iter(tdata.DataLoader(trs[-1], batch_size=batches, drop_last=False, collate_fn=utils.nn.identity)))
	# model = d['model'].to(utils.nn.device)
	# errs = test(model, test_in, test_out)
	# plt.plot(errs)
	# for j in range(5):
	# 	# for i in range(5):
	# 		i=4
	# 		d = load_model(f'Models/resnet_parallel_all/model_checkpoint_batch_{j}_epoch_{i}.pickle')
	# 		model = d['model'].to(utils.nn.device)
	# 		errs = test(model, test_in, test_out)
	# 		plt.plot(errs, label=f'epoch {i} batch {j}')
	# 		# del model
	plt.axis((None, None, -1.5, 1.5))
	plt.legend()
	# import CatBoost
	# model = CatBoost.CatBoost()
	model = d['model'].to(utils.nn.device)
	errs = test(model, test_in, test_out)
	print(errs.mean())
	print(torch.tensor(list(filter(lambda x: x > 0, errs))).mean())
	plt.show()
	r2score = R2Score(num_outputs=utils.data.out_len, multioutput='raw_values').to(utils.nn.device)
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