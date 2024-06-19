import torch, torch.nn as nn
import utils.nn

class AverageEnsemble(nn.Module):
	def __init__(self, models):
		super().__init__()
		# nn.AvgPool1d(kernel_size=368)
		self.models = utils.nn.ParallelModuleList(nn.ModuleList(models))
		self.n_models = len(models)
	def forward(self, x):
		pred = self.models(x).reshape(x.shape[0], 368, self.n_models)
		return pred.mean(dim=2)