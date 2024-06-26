import polars as pl, xarray as xr
import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap
from matplotlib.animation import FuncAnimation, PillowWriter
import matplotlib, tqdm
from mpl_toolkits.mplot3d import Axes3D
import psycopg2.pool as psycopg2_pool
from utils.dset import *
import utils.preprocess as pp
import torch, numpy as np, dill, cloudpickle, sys, glob
from torchmetrics.regression import R2Score
import matplotlib.ticker as mticker
file = r'ClimSim_low-res_grid-info.nc'
grid = xr.open_dataset(file,engine='netcdf4')

DEVICE = 'cuda'

def load_model(path):
	sys.path.append(r'I:\Licenta\impl')
	sys.path.append(r'I:\Licenta\impl\Models\MLP')
	sys.path.append(r'I:\Licenta\impl2')
	try:
		model = dill.load(open(path, 'rb'))
	except:
		try:
			model= cloudpickle.load(open(path, 'rb'))
		except:
			try:
				model = torch.load(path)
			except:
				print('error', path)
				return None
	if isinstance(model, dict):
		d = model
		# print(d.keys())
		model = d['model']
	return model

def map_plot(errs, levels):
	matplotlib.rcParams.update({'font.size':15})
	fig, ax = plt.subplots() #layout='constrained')
	fig.set_size_inches(6.5, 2.5)
	m = Basemap(ax=ax, projection='robin',lon_0=0,resolution='c')
	x,y = m(grid.lon,grid.lat)
	m.drawcoastlines(linewidth=0.5)
	errs = np.clip(errs, -250, 1)
	plot = plt.tricontourf(x,y,errs,cmap='viridis',levels=50, vmin=-250, vmax=1)

	fig.colorbar(plot, ax=ax,
				ticks=[-i for i in reversed(range(0, 260, 50))], extend='min')
	return plot.levels


def get_errs(model, dset):
	model.eval()
	model.to(DEVICE)
	with torch.no_grad():
		losses = []
		r2score = R2Score(num_outputs=368, multioutput='raw_values') #.to(DEVICE)
		ldata = int(dset[0].shape[0]//384)
		# print(ldata)
		for i in range(384):
			din, dout = dset
			din, dout = din[[i + j*384 for j in range(ldata)], :], dout[[i + j*384 for j in range(ldata)], :]
			din, dout = torch.tensor(din, dtype=torch.float64).to(DEVICE), torch.tensor(dout, dtype=torch.float64).to(DEVICE)
			prediction = model(din)
			v_out = pp.preprocess_functions['standardisation']['norm'](pp.preprocess_functions[PREPROC]['denorm'](dout.cpu().numpy()))
			pred = pp.preprocess_functions['standardisation']['norm'](pp.preprocess_functions[PREPROC]['denorm'](prediction.cpu().numpy()))
			losses.append(r2score(torch.tensor(pred), torch.tensor(v_out)).cpu().numpy().mean().item())
	return losses

ldata = 60 #60
PREPROC = 'none'
dset = get_splits(PREPROC, dset_class=TimestepSQLDataset, fraction=0.05)[0].__getitems__(range(ldata))

# pbar = tqdm.tqdm(glob.glob('../models/*'))
# lvls = None
# for path in pbar:
# 	# try:
# 		if 'gmm' in path:
# 			continue
# 		name = path.split('\\')[-1].split('.')[0]
# 		if name not in ['model-resnet6-----']:
# 			continue
# 		pbar.set_postfix_str(f"{name}")

# 		model = load_model(path)
# 		errs = get_errs(model, dset)

# 		lvls = map_plot(errs, lvls)
# 		plt.title(name)
# 		plt.show()
# 		# exit(0)
# 		# plt.savefig(f'figures2/{name}.png')
# 		plt.close()
# 	# except:
# 	# 	print('error', path)
# # print(dset[0].shape)
# exit(0)
torch.set_default_dtype(torch.float64)
import utils.preprocess as pp
r2score = R2Score(num_outputs=368).to('cpu')
r2s = np.zeros(384)

model = cloudpickle.load(open("../impl/gmm_model.pickle", 'rb'))
for i in range(384):
	din, dout = dset
	din, dout = din[[i + j*384 for j in range(ldata)], :], dout[[i + j*384 for j in range(ldata)], :]
	r2tot=0
	# for i in tqdm.trange(0, 60, 10):
	# 	valid_in, valid_out = dset.__getitems__(range(i, (i+10)))
	prediction = model.predict(
			np.array([i for i in range(556)]), din)

	v_out = pp.preprocess_standardisation(dout)
	pred = pp.preprocess_standardisation(prediction)
	v_out, pred = torch.tensor(v_out, dtype=torch.float64), torch.tensor(pred, dtype=torch.float64)
	# preds.append(pred)
	r2 = r2score(pred, v_out)
	r2tot += r2.cpu().numpy()
	r2s[i] = r2tot / 6
map_plot(r2s, None)
plt.title('GMM')
plt.imsave('figures/gmm.png')
plt.show()