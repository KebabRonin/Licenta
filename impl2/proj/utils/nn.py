import numpy as np, torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

def identity(x: tuple[np.ndarray, np.ndarray]):
	return torch.tensor(x[0]), torch.tensor(x[1])

class Printer(torch.nn.Module):
	def __init__(self, name=''):
		super().__init__()
		self.name = name
	def forward(self, x):
		print(self.name, x.shape)
		return x
class MeanPredictor(torch.nn.Module):
	def __init__(self):
		super().__init__()
	def forward(self, x):
		return torch.zeros(x.shape[0], 368, device=x.device)
class SkipConnection(torch.nn.Module):
	dim = 1
	def __init__(self, layers, dim=1):
		super().__init__()
		self.actual = layers
		self.dim = dim

	def forward(self, x):
		pred = self.actual(x)
		return torch.concat([pred, x], dim=self.dim)

class ParallelModuleList(torch.nn.Module):
	def __init__(self, models):
		super().__init__()
		self.models = models

	def forward(self, x):
		out = torch.concat([layer(x) for layer in (self.models)], dim=1)
		return out
