import numpy as np
import matplotlib.pyplot as plt
import glob, torch, tqdm
from my_test import *
if __name__ == '__main__':
	a = []
	errs = []
	normalisation='+mean/std'
	dset = TimestepSQLDataset(norm_method=normalisation)
	splits = get_splits()
	trs = tdata.random_split(dset, splits, generator=torch.Generator().manual_seed(50))
	dloader = DataLoader(
		trs[-2],
		batch_sampler=tdata.BatchSampler(tdata.SequentialSampler(trs[-2]), batch_size=20, drop_last=False),
		collate_fn=identity,
		num_workers=1,
		persistent_workers=False,
		drop_last=False
		# pin_memory=True, # I already send the tensors to cuda, so this might be useless
		# pin_memory_device='cuda',
		# prefetch_factor=4,
	)
	ins, outs = next(iter(dloader))
	fig, ax = plt.subplots()
	for x in tqdm.tqdm(glob.glob('*.pt') + glob.glob('fmodel/*')):
		try:
			model = torch.load(x)
			if isinstance(model, dict):
				d = model
				model=d.pop('model')
				x = {'fname':x, **d}
			s = str(model)

			for idx,  m in enumerate(a):
				if s == m[1]:
					x = str(idx) + x
					m[0].append(x)
					break
			else:
				a.append(([str(len(a)) + x], s))
			validate_model(model, ins, outs, interactive=True, modelname=x, ax=ax) #, ids)
		except:
			errs.append(x)

	print("Models:", len(a))
	print("Errs:", len(errs), errs)

	ax.legend(loc='upper left', bbox_to_anchor=(1.05, 1),
			ncol=2, borderaxespad=0)
	fig.subplots_adjust(right=0.55)
	fig.suptitle('Right-click to hide all\nMiddle-click to show all',
				va='top', size='large')

	leg = interactive_legend()
	plt.axis((None, None, -1.5, 1.5))
	plt.show()

	for x in a:
		print(x[1])
		print(x[0], sep='\n')
		input()