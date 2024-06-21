import utils.nn
import utils.dset
import matplotlib.pyplot as plt
import torch.utils.data as tdata
from torchmetrics.regression import R2Score
import torch
import tqdm
# print(utils.data.in_vars[361])
# for i in range(5):
# 	a = utils.SQLDataset.CustomSQLDataset("none")[i][1]
# 	print(a.shape)
# 	plt.plot(a.squeeze())
# plt.show()


def valid(model, ins, outs):
	r2score = R2Score(num_outputs=utils.data.out_len).to(utils.nn.device)
	ins, outs = ins.to(utils.nn.device), outs.to(utils.nn.device)
	model.eval()
	with torch.no_grad():
		predictions = model(ins)
	loss_value = r2score(predictions, outs).item()
	# ins, outs = ins.to('cpu'), outs.to('cpu')
	return loss_value

def train(model, batch_dloader, optimizer, loss):
	model.train()
	model.to(utils.nn.device)
	losses = []
	pbar = tqdm.tqdm(batch_dloader, position=2, desc="Minibatches", ncols=100)
	for features, targets in pbar:
		features, targets = features.to(utils.nn.device), targets.to(utils.nn.device)
		optimizer.zero_grad()
		predictions = model(features)
		loss_value = loss(predictions, targets)
		loss_value.backward()
		losses.append(loss_value.item())
		pbar.set_postfix_str(f"{loss_value.item()}")
		optimizer.step()
	return losses