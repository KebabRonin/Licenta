import matplotlib.pyplot as plt
import utils.preprocess as pp, utils.data as ud, utils.dset as ds, numpy as np, utils.nn as nns
from torchmetrics.regression import R2Score
import sys, dill, cloudpickle, torch, glob, os, tqdm

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
		for artist in self.legend.texts + self.legend.legend_handles:
			artist.set_picker(10) # 10 points tolerance

		self.fig.canvas.mpl_connect('pick_event', self.on_pick)
		self.fig.canvas.mpl_connect('key_press_event', self.on_click)

	def _build_lookups(self, legend):
		labels = [t.get_text() for t in legend.texts]
		handles = legend.legend_handles
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

	def build_plot(self):
		# plt.show()
		import matplotlib
		# print(matplotlib.rcParams.keys())
		matplotlib.rcParams.update({'font.size':15})
		fig, ax = plt.subplots()
		fig.set_size_inches(15, 7)
		fig.set_layout_engine('constrained')
		ax.clear()
		# plt.figure(figsize=(368 + 100, 500))
		loaded_artists = []
		for artist in self.lookup_artist.values():
			if artist.get_visible() and artist not in loaded_artists:
				ax.plot(artist.get_ydata(), label=artist.get_label().split('|')[1])
				loaded_artists.append(artist)
		anames = [a.get_label().split('|')[0] for a in loaded_artists]

		# dout = dill.load(open(f'rezs/{datasrc}/dout.dill', 'rb'))
		# r2 = R2Score(368, multioutput='raw_values')
		# data = torch.stack([dill.load(open(f'rezs/{datasrc}/{a}.dill','rb')) for a in tqdm.tqdm(anames)])

		# print(data.shape)
		# data = data.mean(0)
		# r2s = r2(data, dout)
		# r2mean = dill.load(open(f'{datasrc}_scores.dill', 'rb'))['mean_predictor'][0]
		# for i in range(368):
		# 	if r2mean[i] > r2s[i]:
		# 		r2s[i] = r2mean[i]

		# ax.plot(r2s, label='avg ensamble', linestyle='dashed',color='black', linewidth=2.5)
		# print("Ensamble Nice R2: ", r2s.mean())

		# for j in range(368):
		# 	order = sorted(anames, key=lambda x: r2s[x][0][j].item(), reverse=True)
		# 	# if r2s[order[0]][0][j] <= 0:
		# 	# 	continue
		# 	data[:, j] = tot[anames.index(order[0]), :, j]
		# ax.plot(r2(data, dout), label='bpt ensamble', linestyle='dashed',color='black', linewidth=2.5)
		# print("Ensamble R2: ", r2(data, dout).numpy().mean())

		data = dill.load(open('en_r2-v2softmax.dill', 'rb')).cpu().numpy()
		data2 = dill.load(open('en_r2-v3softmax.dill', 'rb')).cpu().numpy()
		data3 = np.zeros_like(data)
		meansr2 = loaded_artists[anames.index('mean_predictor')].get_ydata()
		for i in range(368):
			if data2[i] <= 0:
				data3[i] = meansr2[i]
			else:
				data3[i] = data[i]

		ax.plot(data, label='weighted ensamble', linestyle='dashed',color='black', linewidth=2.5)
		print("Ensamble R2: ", data.mean())
		print("Ensamble Nice R2: ", data3.mean())
		if len(loaded_artists) < 6:
			ax.legend(loc='lower right')
		# plt.yscale('symlog')
		ax.set_xticks(*ud.out_ticks)
		ax.set_xlabel('Target variable', weight='bold')
		ax.set_ylabel('R2 Score', weight='bold')
		ax.axis((0, 367, -1.2, 1.2))
		ax.grid()
		fig.show()

	def on_click(self, event):
		if event.key == 'h':
			visible = False
		elif event.key == 'g':
			visible = True
		elif event.key == '`':
			self.build_plot()
			return
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

def plot_values():
	sdevh, sdevlo = pp.preprocess_standardisation(ud.out_std_dev[np.newaxis, :]+ud.out_means[np.newaxis, :]).squeeze(), pp.preprocess_standardisation(-ud.out_std_dev[np.newaxis, :]+ud.out_means[np.newaxis, :]).squeeze()
	mn, mx = pp.preprocess_standardisation(ud.out_mins[np.newaxis, :]).squeeze(), pp.preprocess_standardisation(ud.out_maxs[np.newaxis, :]).squeeze()
	# plt.plot(sdev.squeeze(), label='sdev')
	plt.fill_between([i for i in range(368)], sdevh, sdevlo, color='red', alpha=0.2, label='std_dev')
	# plt.plot([0] * 368, label='mean', color='black')
	plt.plot(mn.squeeze(), label='min')
	plt.plot(mx.squeeze(), label='max')
	# plt.fill_between([i for i in range(368)], mn.squeeze(), mx.squeeze(), color='blue', alpha=0.2, label='value range')
	plt.legend()
	plt.yscale('symlog')
	plt.xticks(*ud.out_ticks)
	plt.axis((0, 367, None, None))
	plt.grid()
	plt.show()

def plot_model(model_path):
	d = dill.load(open(model_path, 'rb'), ignore=True)
	# print(d)

	intervals = list(map(len, d['train_losses']))
	prev = 0
	for i in range(len(intervals)):
		intervals[i] += prev
		prev += intervals[i]
	if len(d['losses']) != len(intervals):
		intervals = [0] + intervals
	plt.plot(d['losses'])
	plt.title(model_path)
	plt.show()
	# plt.plot(intervals, d['losses'])
	# # for x in d['train_losses']:
	# # 	plt.plot(x)
	# plt.plot(sum(d['train_losses'], []))
	# plt.show()

torch.set_default_dtype(torch.float64)
# DONT FORGET -5 TEST -3 VALID -4 VALID2
def generate_lines(nmet = 'standardisation'):
	sys.path.append(r'I:\Licenta\impl')
	sys.path.append(r'I:\Licenta\impl\Models\MLP')
	sys.path.append(r'I:\Licenta\impl2')
	with torch.no_grad():
		ddddd = ds.get_splits(nmet, dset_class=ds.SQLDataset, fraction=0.02)
		print(len(ddddd))
		din, dout = ddddd[-4].__getitems__(range(19_200)) #-3 if datasrc == 'valid' else -5
		din, dout = torch.tensor(din, device=nns.device), torch.tensor(dout, device=nns.device)
		din, dout = din.to(nns.device), dout.to(nns.device)

		errs = []
		# dd = dict()
		# dd = dill.load(open('nice_r2_scores.dill', 'rb'))
		# vsc = dill.load(open('valid_scores.dill', 'rb'))
		# meanns = pp.preprocess_functions[nmet]['norm'](ud.out_means[np.newaxis, :]).squeeze()
		# r2score = R2Score(num_outputs=368, multioutput="raw_values").to(nns.device)
		total = len(glob.glob('I:/Licenta/models/*'))
		for i, x in enumerate(glob.glob('I:/Licenta/models/*')):
			name = x[len('I:/Licenta/models/'):].split('.')[0]
			if name not in ['model-resnet6-----', 'kan-3', 'kan']:
				print(f'{i}/{total}', 'skipping other norm', x)
				continue
			if 'gm' in x or 'kan-4' in x:
				print(f'{i}/{total}', 'skipping', x)
				continue
				rez = model.predict(din)
			try:
				model = dill.load(open(x, 'rb'))
			except:
				try:
					model= cloudpickle.load(open(x, 'rb'))
				except:
					try:
						model = torch.load(x)
					except:
						print(f'{i}/{total}', 'error', x)
						errs.append(x)
						continue
			if isinstance(model, dict):
				d = model
				# print(d.keys())
				model = d['model']
			# print(name, model, '\n===================================')
			print(f'{i}/{total}', name, nmet)
			model.to(nns.device)
			rez = model(din)
			del model
			rez2 = pp.preprocess_functions[nmet]['denorm'](rez.cpu().numpy())
			rez2 = pp.preprocess_standardisation(rez2)
			dill.dump(torch.tensor(rez), open(f'rezs/valid2/{name}.dill', 'wb'))
			# del model
			# for j in range(368):
			# 	# print(vsc[name][j])
			# 	if vsc[name][0][j] <= 0:
			# 		rez[:, j] = meanns[j]
			# dd[name] = (r2score(rez, dout).cpu().numpy(), nmet)
			# print(name, dd[name][0].mean())
			# dill.dump(dd, open('nice_r2_scores.dill', 'wb'))
			# plt.plot(dd[name][0])
			# plt.axis((0, 367, -0.5, 1.2))
			# plt.xticks([0, 60, 120, 180, 240, 300, 360], ['ptend_t', 'ptend_q0001', 'ptend_q0002', 'ptend_q0003', 'ptend_u', 'ptend_v', 'cam_out'])
			# plt.grid()
			# plt.title(name + nmet)
			# plt.show()

		print(errs)

def list_models():
	sys.path.append(r'I:\Licenta\impl')
	sys.path.append(r'I:\Licenta\impl\Models\MLP')
	sys.path.append(r'I:\Licenta\impl2')
	with torch.no_grad():
		errs = []
		# dd = dict()
		for i, x in enumerate(glob.glob('I:/Licenta/models/*')):
			name = x[len('I:/Licenta/models/'):].split('.')[0]
			try:
				model = dill.load(open(x, 'rb'))
			except:
				try:
					model= cloudpickle.load(open(x, 'rb'))
				except:
					try:
						model = torch.load(x)
					except:
						errs.append(x)
						continue
			if isinstance(model, dict):
				d = model
				model = d['model']
			print(name, model, '\n===================================')

		print(errs)

def plot_lines():
	d = dill.load(open(f'{datasrc}_scores.dill', 'rb'))
	fig, ax = plt.subplots()
	ls = []
	ns = sorted(d.keys(), key=lambda x: np.mean(d[x][0])) #['gmm', ]
	for mname in sorted(d.keys(), key=lambda x: np.mean(d[x][0])):
		ax.plot(d[mname][0], label=f'{mname}|{d[mname][2]}')
		if mname in ns:
			ls.append(d[mname][0])
		print(f'{mname:<32} | {d[mname][2]:<20} | {np.mean(d[mname][0]):>0.10f}')


	ax.legend(loc='upper left', bbox_to_anchor=(1.05, 1),
			ncol=2, borderaxespad=0)
	fig.subplots_adjust(right=0.55)
	fig.suptitle('"h" to hide all, "g" to show all, "`" to save current graph',
				va='top', size='large')
	leg = interactive_legend()
	plt.axis((0, 367, -0.5, 1.2))
	plt.xticks([0, 60, 120, 180, 240, 300, 360], ['ptend_t', 'ptend_q0001', 'ptend_q0002', 'ptend_q0003', 'ptend_u', 'ptend_v', 'cam_out'])
	plt.grid()
	# plt.legend()
	plt.show()


"""
x cb7
x cb8
x gmm: gmm 200
x mlp-bottleneck-32batch-sql, mlp-bottleneck-32batch-timestep, mlp-bottleneck: mlp-3-2-2-2-3; unit = 556+368 - st
model-sss: LeakyReLU 6 x 2skipconn weird???
model_no_layernorm_mine_656 LeakyReLU 6 x 2skipconn weird??
model_layernorm_mine_356 LeakyReLU 6 x 2skipconn weird?
x model-optuna-0-08xx: optuna w/ skipconns
model-0: 6 x weird  skipconns
model 6 x 5resblock + 69 Tanh
model-optuna-smoothl1, model_resnet5_worse, model: ReLU 6 x 5resblock + 69 Tanh
model-resnet6-----: SiLU 6 x 6resblock + 1resblock										- minmax100
model_resnet6 ReLU 6 x 6resblock + 69 Tanh
resnet-parallel-all: 92 (4 outputs per net) x 2resblock 128 128 ReLU
resnet-parallel: 6 x 7resblock 512 512 ReLU + 2resblock
resnet-simple: 1 x 7resblock ReLU 1024 1024
x cnn: UNet - maybe see iter 5?
kan-3, kan-5: kan-1112-1112
kan-6: kan-1113-512
kan-7: kan-2048-512
---
kan-5,6 standardisation good
kan-3 minmax100 okish
kan-7 standardisation bad
---TRASH
kan-2:   kan-30
kan   kan-30 - see more about this minmax100
kan-4: nothing kan, trash
"""



# d=dict()
# for x in glob.glob('I:/Licenta/models/*'):
# 	d[name] = [0 for _ in range(368)]
# dill.dump(d, open('final_scores.dill', 'wb'))
# for n in ds.preprocess_functions.keys():
# 	generate_lines(n)
# generate_lines('minmax100')
# gen_r2s('valid2')
datasrc = 'test'
plot_lines()
# ddddd = ds.get_splits('standardisation', dset_class=ds.SQLDataset, fraction=0.02)
# print(len(ddddd))
# _, dout = ddddd[-3].__getitems__(range(19_200))
# dout = torch.tensor(dout, dtype=torch.float64)
# dill.dump(dout, open('rezs/valid/dout.dill', 'wb'))
# list_models()
# d = dill.load(open('fscores.dill', 'rb'))
# dds = list(d.keys())
# for m in dds:
# 	k = input(f'rename {m} to: ')
# 	d[m] = (*d[m], k)
# dill.dump(d, open('final_scores2.dill', 'wb'))
# for n in glob.glob('I:/Licenta/models/*'):
# 	try:
# 		plot_model(n)
# 	except:
# 		pass




# plot_model(r'Models\cnn\model_checkpoint_batch_0_epoch_19.pickle')
# from torch.utils.data import DataLoader
# import torch
# vv_in, vv_out = next(iter(DataLoader(ds.get_splits('none', dset_class=ds.TimestepSQLDataset, fraction=0.02)[0], batch_size=1, drop_last=False, collate_fn=nns.identity)))
# print(vv_in.shape, vv_out.shape)
# vv = torch.concat([vv_in, vv_out], dim=1)
# print(vv.shape)
# # while True:
# # 	c = input('var:')
# # 	if c == 'q':
# # 		break
# # 	idx = ud.all_vars.index(c)
# # 	plt.hist(vv[:, idx])
# # 	plt.title(c)
# # 	plt.show()

# begin, end = ud.all_vars.index('ptend_q0002_0'), ud.all_vars.index('ptend_q0002_59')+1
# beginn, endd = (ud.all_means[begin:end] - ud.all_std_dev[begin:end]).min(), (ud.all_means[begin:end] + ud.all_std_dev[begin:end]).max()
# print(beginn, endd)
# # plt.hist(vv[:, begin], bins=100, range=range)
# # img = np.concatenate([np.histogram(vv[:, i], bins=100)[0][..., np.newaxis]/vv.shape[0] for i in range(begin, end)], axis=-1)
# # plt.imshow(img)
# # plt.colorbar()
# # plt.show()
# for i in range(begin, end):
# 	plt.hist(vv[:, i], bins=50)
# 	plt.show()