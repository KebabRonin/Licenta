import numpy as np
import matplotlib.pyplot as plt
import glob, torch, tqdm, dill
from utils.dset import *
from utils.nn import *
from utils.preprocess import *
import testt
from torch.utils.data import DataLoader
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
import torchmetrics.regression, utils.nn, utils.data
def teast(model, ins, outs):
	print('ad')
	print(ins, outs, utils.nn.device)
	# ins, outs = ins.cpu(), outs.cpu()
	# ins, outs = ins.to(utils.nn.device), outs.to(utils.nn.device)
	print('ad')
	r2score = torchmetrics.regression.R2Score(num_outputs=utils.data.out_len, multioutput='raw_values') #.to(utils.nn.device)
	mse = torch.nn.MSELoss().to(utils.nn.device)
	model.eval()
	with torch.no_grad():
		prediction = model(ins)
		loss_values = r2score(prediction, outs).cpu()
	return loss_values

ddd = dict()
if __name__ == '__main__':
	a = []
	errs = []
	trs = get_splits(norm_method='none', dset_class=TimestepSQLDataset, fraction=0.05)
	dloader = DataLoader(
		trs[-2],
		batch_sampler=tdata.BatchSampler(tdata.SequentialSampler(trs[-2]), batch_size=30, drop_last=False),
		collate_fn=identity,
		num_workers=1,
		persistent_workers=False,
		drop_last=False
		# pin_memory=True, # I already send the tensors to cuda, so this might be useless
		# pin_memory_device='cuda',
		# prefetch_factor=4,
	)
	inss, outss = next(iter(dloader))
	inss, outss = inss.numpy(), outss.numpy()
	fig, ax = plt.subplots()
	for x in tqdm.tqdm(glob.glob('*.pickle') + [glob.glob(f"{f}/*.pickle")[-1] for f in glob.glob('Models/*')]):
			print(x)
		# try:
			model = dill.load(open(x, 'rb'))
			normalisation = 'standardisation'
			if isinstance(model, dict):
				d = model
				model=d.pop('model')
				normalisation = d['norm_method']
				# x = {'fname':x, **d}
			s = str(model)

			for idx, m in enumerate(a):
				if s == m[1]:
					x = str(idx) + x
					m[0].append(x)
					break
			else:
				x = str(len(a)) + x
				a.append(([x], s))
			model.to(utils.nn.device)
			ins, outs = torch.tensor(preprocess_functions[normalisation]['norm'](inss)) , torch.tensor(preprocess_functions[normalisation]['norm'](outss))
			with torch.no_grad():
				prediction = model(ins)
				loss_values = torchmetrics.regression.R2Score(num_outputs=utils.data.out_len, multioutput='raw_values')(prediction, outs).cpu()
			ddd[x] = loss_values
			# ddd[x] = teast(model, ins, outs) #, ids)
			print(ddd[x])
			ax.plot(ddd[x], label=x)
		# except Exception as e:
		# 	errs.append(x)
		# 	print(e)
	dill.dump(ddd, open('r2scores.dill', 'wb'))
	print("Models:", len(a))
	print("Errs:", len(errs), errs)

	ax.legend(loc='upper left', bbox_to_anchor=(1.05, 1),
			ncol=2, borderaxespad=0)
	fig.subplots_adjust(right=0.55)
	fig.suptitle('Right-click to hide all\nMiddle-click to show all',
				va='top', size='large')

	for x in a:
		print(x[1])
		print(x[0], sep='\n')

	leg = interactive_legend()
	plt.axis((0, 367, -0.5, 1.2))
	plt.xticks([0, 60, 120, 180, 240, 300, 360], ['ptend_t', 'ptend_q0001', 'ptend_q0002', 'ptend_q0003', 'ptend_u', 'ptend_v', 'cam_out'])
	plt.grid()
	plt.show()
