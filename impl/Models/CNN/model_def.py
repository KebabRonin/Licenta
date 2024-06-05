import numpy as np
import torch, polars as pl
import torch.nn as nn
import torch.nn.functional as F

in_len = 556
out_len = 368

class ResBlock(nn.Module):
	def __init__(self, layers):
		super().__init__()
		self.actual = layers

	def forward(self, x):
		return torch.concat([self.actual(x), x], dim=1)


class CNN(nn.Module):
	def __init__(self):
		super().__init__()
		self.l_60_models = nn.ModuleList([nn.Sequential(
			nn.Conv1d(in_channels=60, out_channels=20, kernel_size=(2, )),
			nn.LeakyReLU(0.15),
			nn.Conv1d(in_channels=60, out_channels=20, kernel_size=(2, )),
			nn.LeakyReLU(0.15),
			nn.Conv1d(in_channels=60, out_channels=20, kernel_size=(2, )),
			nn.LeakyReLU(0.15),
			nn.Conv1d(in_channels=60, out_channels=20, kernel_size=(2, )),
			nn.Linear(in_len, 512),
			nn.LeakyReLU(0.15),
			nn.Linear(512, 512),
			nn.LeakyReLU(0.15),
			nn.Linear(512, 512),
			nn.LeakyReLU(0.15),
			nn.Linear(512, 512),
			nn.LeakyReLU(0.15),
			nn.Linear(512, 512),
			nn.LeakyReLU(0.15),
			nn.Linear(512, 60),
			# nn.SELU(),
			# nn.Linear(60),
		) for _ in range (6)]) # nr of 60-level vars
		self.l_vars = nn.Sequential(
			nn.Linear(in_len, 512),
			nn.SELU(),
			nn.Linear(512, 256),
			nn.SELU(),
			nn.Linear(256, 8),
		)

	def forward(self, x):
		# # v1:
		# x = self.linear_relu_stack(x)
		# out = torch.concat([self.o_60(x), self.o_scalar(x)], dim=1)
		# v2:
		out = torch.concat([layer(x[i*60]) for i, layer in enumerate(self.l_60_models)], dim=1)
		return out

model = CNN()

if __name__ == '__main__':
	print(model)
