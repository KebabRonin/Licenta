import numpy as np, torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def identity(x: tuple[np.ndarray, np.ndarray]):
	return torch.tensor(x[0]), torch.tensor(x[1])

class SkipConnection(torch.nn.Module):
	def __init__(self, layers):
		super().__init__()
		self.actual = layers

	def forward(self, x):
		return torch.concat([self.actual(x), x], dim=1)

class ParallelModuleList(torch.nn.Module):
	def __init__(self, models):
		super().__init__()
		self.models = models

	def forward(self, x):
		out = torch.concat([layer(x) for layer in (self.models)], dim=1)
		return out
