import torch, dill, tqdm, glob, matplotlib.pyplot as plt
from torchmetrics.regression import R2Score

class EnsembleModel(torch.nn.Module):
	def __init__(self, nmodels):
		super().__init__()
		self.models = torch.nn.ModuleList([torch.nn.Linear(nmodels, 1) for _ in range(368)])

	def forward(self, x):
		# x.shape = models x samples x features
		resp = [
			self.models[i](x[:, i, :].squeeze()) for i in range(len(self.models))
			]
		# print(f"{resp[0].shape=}")
		return torch.concatenate(resp, dim=1)
class Printer(torch.nn.Module):
	def __init__(self):
		super().__init__()

	def forward(self, x):
		# x.shape = models x samples x features
		print(f"{x.shape=}")
		return x


class EnsembleModel2(torch.nn.Module):
	def __init__(self, nmodels):
		super().__init__()
		self.models = torch.nn.ModuleList([torch.nn.Sequential(torch.nn.Linear(nmodels, nmodels), torch.nn.Softmax(dim=1)) for _ in range(368)])

	def forward(self, x):
		# x.shape = models x samples x features
		resp = [
			self.models[i](x[:, i, :].squeeze()) for i in range(len(self.models))
			]
		# print(resp[0].shape)
		resp = torch.stack(resp, dim=1)
		# print(resp.shape, x.shape)
		# print(f"{resp[0].shape=}")
		return torch.mul(resp, x).sum(dim=2)


torch.set_default_dtype(torch.float64)



def train_model():
	dout = torch.tensor(dill.load(open('rezs/valid/dout.dill', 'rb')), device='cuda')
	dout_test = torch.tensor(dill.load(open('rezs/valid2/dout.dill', 'rb')), device='cuda')
	print(f"{dout.shape=}")
	r2score = R2Score(368, multioutput='raw_values').to('cuda')
	data = torch.stack([torch.tensor(dill.load(open(a,'rb')), device='cuda') for a in tqdm.tqdm(glob.glob(f'rezs/valid/*'))], dim=-1).to('cuda')
	data_test = torch.stack([torch.tensor(dill.load(open(a,'rb')), device='cuda') for a in tqdm.tqdm(glob.glob(f'rezs/valid2/*'))], dim=-1).to('cuda')
	print(f"{data.shape=}")
	enm = EnsembleModel2(data.shape[-1]).to('cuda')
	# enm = dill.load(open('ensamble.dill', 'rb'))

	optim = torch.optim.RAdam(enm.parameters(), lr=0.0001, weight_decay=0.01)
	loss_fn = torch.nn.MSELoss()
	losses = []
	tlosses = []
	pbar = tqdm.trange(2_500)
	for e in pbar:
		optim.zero_grad()
		pred = enm(data)
		# print(f"{pred.shape=}")
		loss = loss_fn(pred, dout)
		loss.backward()
		optim.step()
		losses.append(loss.item())
		with torch.no_grad():
			pred = enm(data_test)
			loss = r2score(pred, dout_test)
			tlosses.append(loss.mean().item())
			if e % 30 == 0:
				plt.figure()
				plt.plot(loss.cpu().numpy())
				plt.axis((0, 367, -1.2, 1.2))
				plt.grid()
				plt.xticks([0, 60, 120, 180, 240, 300, 360], ['ptend_t', 'ptend_q0001', 'ptend_q0002', 'ptend_q0003', 'ptend_u', 'ptend_v', 'cam_out'])
				plt.savefig('errors.png')

		pbar.set_postfix_str(f"{losses[-1]:.10f}|{tlosses[-1]:.20f}")
		if len(tlosses) > 8 and tlosses[-1] < tlosses[-7]:
			break

	dill.dump(enm, open('ensamble_v4softmax.dill', 'wb'))

	plt.close('all')
	plt.plot(losses, label='train')
	plt.plot(tlosses, label='test')
	plt.show()

def test_model():
	dout_test = torch.tensor(dill.load(open('rezs/valid2/dout.dill', 'rb'))).to('cuda')
	data_test = torch.stack([torch.tensor(dill.load(open(a,'rb')), device='cuda') for a in tqdm.tqdm(glob.glob(f'rezs/valid2/*'))], dim=-1).to('cuda')
	enm = dill.load(open('ensamble_v4softmax.dill', 'rb'))
	r2score = R2Score(368, multioutput='raw_values').to('cuda')
	with torch.no_grad():
		pred = enm(data_test)
		r2 = r2score(pred, dout_test)
		print(r2.mean())
		plt.plot(r2.cpu().numpy())
		plt.axis((0, 367, -1.2, 1.2))
		plt.grid()
		plt.xticks([0, 60, 120, 180, 240, 300, 360], ['ptend_t', 'ptend_q0001', 'ptend_q0002', 'ptend_q0003', 'ptend_u', 'ptend_v', 'cam_out'])
		plt.show()
	dill.dump(r2, open('en_r2-v44softmax.dill', 'wb'))

# train_model()
test_model()