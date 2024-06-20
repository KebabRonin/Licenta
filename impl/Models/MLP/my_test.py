import polars as pl, numpy as np, torch
# from torcheval.metrics.functional import r2_score
from torchmetrics.regression import R2Score
from torch.nn import L1Loss, MSELoss
# from model_def import model
from my_utils import *
import sys, time, json
sys.stdout.reconfigure(encoding='utf-8')
from mpl_toolkits.basemap import Basemap
import xarray as xr
def interactive_legend(ax=None):
    if ax is None:
        ax = plt.gca()
    if ax.legend_ is None:
        ax.legend()

    return InteractiveLegend(ax.get_legend())

class InteractiveLegend(object):
    def __init__(self, legend):
        self.legend = legend
        self.fig = legend.axes.figure

        self.lookup_artist, self.lookup_handle = self._build_lookups(legend)
        self._setup_connections()

        self.update()

    def _setup_connections(self):
        for artist in self.legend.texts + self.legend.legendHandles:
            artist.set_picker(10) # 10 points tolerance

        self.fig.canvas.mpl_connect('pick_event', self.on_pick)
        self.fig.canvas.mpl_connect('button_press_event', self.on_click)

    def _build_lookups(self, legend):
        labels = [t.get_text() for t in legend.texts]
        handles = legend.legendHandles
        label2handle = dict(zip(labels, handles))
        handle2text = dict(zip(handles, legend.texts))

        lookup_artist = {}
        lookup_handle = {}
        for artist in legend.axes.get_children():
            if artist.get_label() in labels:
                handle = label2handle[artist.get_label()]
                lookup_handle[artist] = handle
                lookup_artist[handle] = artist
                lookup_artist[handle2text[handle]] = artist

        lookup_handle.update(zip(handles, handles))
        lookup_handle.update(zip(legend.texts, handles))

        return lookup_artist, lookup_handle

    def on_pick(self, event):
        handle = event.artist
        if handle in self.lookup_artist:

            artist = self.lookup_artist[handle]
            artist.set_visible(not artist.get_visible())
            self.update()

    def on_click(self, event):
        if event.button == 3:
            visible = False
        elif event.button == 2:
            visible = True
        else:
            return

        for artist in self.lookup_artist.values():
            artist.set_visible(visible)
        self.update()

    def update(self):
        for artist in self.lookup_artist.values():
            handle = self.lookup_handle[artist]
            if artist.get_visible():
                handle.set_visible(True)
            else:
                handle.set_visible(False)
        self.fig.canvas.draw()

    def show(self):
        plt.show()

def validate_model(model, ins=None, outs=None, ids=None, interactive=False, normalisation='+mean/std', modelname='', ax=None):
	if ins is None or outs is None:
		dset = CustomSQLDataset(norm_method=normalisation)
		splits = get_splits()
		trs = tdata.random_split(dset, splits, generator=torch.Generator().manual_seed(50))
		dloader = DataLoader(
			trs[-2],
			batch_sampler=tdata.BatchSampler(tdata.SequentialSampler(trs[-2]), batch_size=10_000, drop_last=False),
			collate_fn=identity,
			num_workers=1,
			persistent_workers=False,
			drop_last=False
			# pin_memory=True, # I already send the tensors to cuda, so this might be useless
			# pin_memory_device='cuda',
			# prefetch_factor=4,
		)
		ins, outs = next(iter(dloader))
		# print("Read in")
	model.eval()

	r2score_true = R2Score(num_outputs=368).to('cuda')
	if interactive:
		r2score = R2Score(num_outputs=368, multioutput="raw_values").to('cuda')
		maescore = L1Loss()
		print(f"Tested at: {time.strftime('%d-%b-%Y | %H:%M:%S')}")
	with torch.no_grad():
		ins, outs = ins.to(DEVICE), outs.to(DEVICE)
		prediction = model(ins)
		# for i in range(27):
		# 	prediction[:, 120+i] = -ins[:, 120+i] / 1200
		# print(prediction[0][:20])
		# print(outs[0][:20])
		tru = r2score_true(prediction, outs)
		if interactive:
			r2 = r2score(prediction, outs)
			mae = maescore(prediction, outs)
			# header=f"{'Name':^15}|{'Actual':^10}|{'Pred':^10}|{'Diff':^10}|{'R2':^10}|*|"
			# print(f"{'&':=^186}")
			# print(*(header for _ in range(3)), sep='')
			# print(f"{'&':=^186}")
			# fstr = "{vname:<15}|{act:>10.5f}|{pred:>10.5f}|{diff:>10.5f}|{r2ll:>10.5f}|*|"
			# for i in range(0, len(out_vars), 3):
			# 	print(*(
			# 		fstr.format(
			# 			vname=out_vars[idx],
			# 			act=outs[0][idx],
			# 			pred=prediction[0][idx],
			# 			diff=outs[0][idx] - prediction[0][idx],
			# 			r2ll=r2[idx],
			# 		) for idx in range(i, i+3) if idx < len(out_vars)
			# 	), sep='')
			# print(f"{'&':=^186}")
			mask = np.ones(len(r2), dtype=bool)
			ok_indices = [idx for idx in range(len(out_vars)) if out_vars[idx] in zeroed_vars]
			mask[ok_indices] = False
			result = r2[mask,...]
			nice= sum(result.tolist()) #list(filter(lambda x: x >= 0, result.tolist())))
			nicer = sum(list(filter(lambda x: x >= 0, result.tolist())))
			notnice= sum(r2.tolist())

			print("MAE	   :", mae.item())
			print("*Nice*  r2:", nice/len(non_zeroed_out_vars))
			print("*Nicer* r2:", nicer/len(non_zeroed_out_vars))
			print("Actual  r2:", notnice/len(out_vars))
			print("True", tru)
			ax.plot(r2.cpu(), label=modelname)
			# plt.show()
			prediction = prediction.cpu()
			outs = outs.cpu()
			# try:
			# 	while True:
			# 		c = input("column name:")
			# 		c = out_vars.index(c)
			# 		plt.plot(prediction[:, c], label='pred')
			# 		plt.plot(outs[:, c], label='actual')
			# 		plt.legend()
			# 		plt.title(out_vars[c])
			# 		plt.show()
			# 		if ids is not None:
			# 			mapplot(c, prediction, outs, ids)
			# except:
			# 	pass
	model.train()
	return tru.item()
def mapplot(c, prediction, outs, ids):
	file = r'C:\Users\KebabWarrior\Desktop\Facultate\ClimSim\grid_info\ClimSim_low-res_grid-info.nc'
	grid = xr.open_dataset(file,engine='netcdf4')
	if ids is not None:
		# print(ids[:20])
		x = grid.lon
		y = grid.lat
		m = Basemap(projection='robin',lon_0=165,resolution='c')
		startt = (384 - ids[0]%384)
		pp = sum([prediction[(st):(st+384), c] for st in range(startt, (nr_rows//384 - 1) * 384, 384)], prediction[startt:(startt+384), c])
		oo = sum([outs[(st):(st+384), c] for st in range(startt, (nr_rows//384 - 1) * 384, 384)], outs[startt:(startt+384), c])
		data = R2Score(num_outputs=384, multioutput="raw_values")(pp, oo).to('cpu')
		print(data.shape)
		m.drawcoastlines(linewidth=0.5)
		x,y = m(grid.lon,grid.lat)
		plot = plt.tricontourf(x,y,data,cmap='viridis',levels=14)
		m.colorbar(plot)
		plt.show()
	# if ids is not None:
	# 	# print(ids[:20])
	# 	x = grid.lon
	# 	y = grid.lat
	# 	m = Basemap(projection='robin',lon_0=165,resolution='c')
	# 	startt = (384 - ids[0]%384)
	# 	pp = sum([prediction[(st):(st+384), :] for st in range(startt, (nr_rows//384 - 1) * 384, 384)], prediction[startt:(startt+384), :])
	# 	oo = sum([outs[(st):(st+384), :] for st in range(startt, (nr_rows//384 - 1) * 384, 384)], outs[startt:(startt+384), :])
	# 	pp = torch.transpose(torch.sum(pp, dim=1), 0, -1)
	# 	oo = torch.transpose(torch.sum(oo, dim=1), 0, -1)
	# 	print(pp.shape, oo.shape)
	# 	data = R2Score(num_outputs=384, multioutput="raw_values")(pp, oo).to('cpu')
	# 	print(data)
	# 	print(data.shape)
	# 	m.drawcoastlines(linewidth=0.5)
	# 	x,y = m(grid.lon,grid.lat)
	# 	plot = plt.tricontourf(x,y,data,cmap='viridis',levels=14)
	# 	m.colorbar(plot)
	# 	plt.show()
if __name__ == '__main__':
	# model = torch.load('model.pt')
	# if isinstance(model, dict):
	# 	model = model['model']
	nr_rows = 10_000
	normalisation = '+mean/std'
	# print(model)
	dset = CustomSQLDataset(norm_method=normalisation)
	splits = get_splits()
	trs = tdata.random_split(dset, splits, generator=torch.Generator().manual_seed(50))
	dloader = DataLoader(
		trs[-2],
		batch_sampler=tdata.BatchSampler(tdata.SequentialSampler(trs[-2]), batch_size=10_000, drop_last=False),
		collate_fn=identity,
		num_workers=1,
		persistent_workers=False,
		drop_last=False
		# pin_memory=True, # I already send the tensors to cuda, so this might be useless
		# pin_memory_device='cuda',
		# prefetch_factor=4,
	)
	ins, outs = next(iter(dloader))
	# valid = pl.read_parquet([f"Dataset/train/v1/train_{i}.parquet" for i in range(40, 51)], n_rows=nr_rows)
	# valid = pl.read_parquet("Dataset/train/v1/train_46.parquet", n_rows=nr_rows)

	# ids = np.array(list(map(lambda x: int(x[6:]), valid['sample_id'])))
	# ins = normalize_subset(valid,in_vars, method="+mean/std")
	# outs = normalize_subset(valid,out_vars, method="+mean/std")
	# ins = torch.tensor(ins.to_numpy(), device='cuda')
	# outs= torch.tensor(outs.to_numpy(), device='cuda')

	fig, ax = plt.subplots()

	import glob
	for m in glob.glob("*resnet5*.pt"):
		try:
			model = torch.load(m)
			if isinstance(model, dict):
				model = model['model']
			print(model)
			validate_model(model, ins, outs, interactive=True, modelname=m, ax=ax) #, ids)
			print(m, 'ok')
		except Exception as e:
			print(m, 'fail', e)

	ax.legend(loc='upper left', bbox_to_anchor=(1.05, 1),
			ncol=2, borderaxespad=0)
	fig.subplots_adjust(right=0.55)
	fig.suptitle('Right-click to hide all\nMiddle-click to show all',
				va='top', size='large')

	leg = interactive_legend()
	plt.axis((None, None, -1.5, 1.5))
	plt.show()
"""MAE	  : 0.2969637971058436
*Nice* r2: 0.19872192730669117
Actual r2: -87.40791329077405"""
# valid = pl.scan_parquet([f"Dataset/train/v1/train_{i}.parquet" for i in range(51)])

# l = valid.select(pl.len()).collect().item()
# each = int(l/20)
# for i in valid.iter_slices(each):
# 	df = valid.slice(i*each, each).cast({v:pl.Float32 for v in (in_vars + out_vars)})
# 	print(df.slice(0, 1).collect())
# 	df.collect()
# 	df.write_parquet("Dataset/train/v3/train_{i}.parquet")