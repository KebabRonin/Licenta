import numpy as np
import torch, polars as pl
import torch.nn as nn
import torch.nn.functional as F
from my_utils import SkipConnection

in_len = 556
out_len = 368

l_widths = [556, 1024, 512, 368]

class MLP(nn.Module):
	def __init__(self):
		super().__init__()
		self.model = nn.Sequential(
			SkipConnection(nn.Sequential(
				nn.Linear(in_len, 1024),
				nn.LayerNorm(1024),
				nn.ReLU(),
				nn.Linear(1024, 512),
				nn.LayerNorm(512),
				nn.ReLU(),
			)),
			nn.Linear(in_len+512, 368),
		)

	def forward(self, x):
		return self.model(x)

model = MLP()

if __name__ == '__main__':
	print(model)
	pytorch_total_params = sum(p.numel() for p in model.parameters())
	print(pytorch_total_params)
