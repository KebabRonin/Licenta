import numpy as np
import matplotlib.pyplot as plt
import glob, torch, tqdm, dill
from my_test import *
ddd = dict()
if __name__ == '__main__':
	a = []
	errs = []
	normalisation='+mean/std'

	fig, ax = plt.subplots()
	# for x in tqdm.tqdm(glob.glob('*.pt') + glob.glob('fmodel/*')):
	norms = ['minmax100', '+mean', 'mean norm', 'minmax10']
	for n, x in enumerate(tqdm.tqdm(glob.glob('model_optuna_12-Jun-2024-13-51-07_minmax100.pt') +
					glob.glob('model_optuna_12-Jun-2024-14-50-25_+mean.pt') +
					glob.glob('model_optuna_12-Jun-2024-15-45-43_mean norm.pt') +
					glob.glob('model_optuna_12-Jun-2024-16-45-07_minmax10.pt'))):
		try:
			model = torch.load(x)
			if isinstance(model, dict):
				d = model
				model=d.pop('model')
				# x = {'fname':x, **d}
			s = str(model)
			normalisation = norms[n]
			print(x, normalisation)

			dset = TimestepSQLDataset(norm_method=normalisation)
			splits = get_splits()
			trs = tdata.random_split(dset, splits, generator=torch.Generator().manual_seed(50))
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
			ins, outs = next(iter(dloader))

			for idx, m in enumerate(a):
				if s == m[1]:
					x = str(idx) + x
					m[0].append(x)
					break
			else:
				x = str(len(a)) + x
				a.append(([x], s))
			validate_model(model, ins, outs, interactive=True, modelname=x, ax=ax, ddd=ddd) #, ids)
		except Exception as e:
			errs.append(x)
			print(s)
			print(e)
	dill.dump(ddd, open('r2scores222.dill', 'wb'))
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
