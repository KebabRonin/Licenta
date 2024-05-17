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
		# self.out = torch.empty(x.shape[0] +

	def forward(self, x):
		# print(x.shape)
		return torch.concat([self.actual(x), x], dim=1)

# l_widths = [768, 640, 512, 640, 640, 360]

class MLP(nn.Module):
	def __init__(self):
		super().__init__()
		hidden_size = 4096
		act = nn.ELU
		# v3:
		self.l_60_models = nn.Sequential(
			ResBlock(nn.Sequential(
				nn.Linear(in_len, hidden_size),
				act(),
				nn.Linear(hidden_size, hidden_size),
				act(),
				nn.Linear(hidden_size, hidden_size),
				act(),
				nn.Linear(hidden_size, hidden_size),
				act(),
			)),
			ResBlock(nn.Sequential(
				nn.Linear(in_len+hidden_size, hidden_size),
				act(),
				nn.Linear(hidden_size, hidden_size),
				act(),
				nn.Linear(hidden_size, hidden_size),
				act(),
				nn.Linear(hidden_size, hidden_size),
				act(),
			)),
			nn.Linear(in_len+2*hidden_size, 368),
		) # nr of 60-level vars
		self.l_vars = nn.Sequential(
			ResBlock(nn.Sequential(
				nn.Linear(in_len, 1024),
				nn.SiLU(),
				nn.Linear(1024, 256),
				nn.LeakyReLU(0.15),
			)),
			nn.Linear(256+in_len, 8),
			# nn.SELU(),
			# nn.Linear(8),
		)
		# self.out = torch.tensor(np.empty(shape=(368,)))

	def forward(self, x):
		# v2:
		out = torch.concat([self.l_60_models(x), self.l_vars(x)], dim=1)
		return out # self.l_60_models(x)

model = MLP()

if __name__ == '__main__':
	print(model)
	pytorch_total_params = sum(p.numel() for p in model.parameters())
	print(pytorch_total_params)

		# v2:
		# self.linear_relu_stack = nn.Sequential(
		# 	# nn.LayerNorm((in_len,)),
		# 	nn.Linear(in_len, l_widths[0]),
		# 	nn.LeakyReLU(negative_slope=0.15),
		# 	nn.Linear(l_widths[0], l_widths[1]),
		# 	nn.LeakyReLU(negative_slope=0.15),
		# 	nn.Linear(l_widths[1], l_widths[2]),
		# 	nn.LeakyReLU(negative_slope=0.15),
		# 	nn.Linear(l_widths[2], l_widths[3]),
		# 	nn.LeakyReLU(negative_slope=0.15),
		# 	nn.Linear(l_widths[3], l_widths[4]),
		# 	nn.LeakyReLU(negative_slope=0.15),
		# 	nn.Linear(l_widths[4], l_widths[5]),
		# 	nn.LeakyReLU(negative_slope=0.15),
		# 	# nn.Linear(l_widths[5], out_len),
		# )
		# self.o_scalar = nn.Sequential(nn.Linear(l_widths[5], 8), nn.ReLU())
		# self.o_60 = nn.Linear(l_widths[5], 360)


		# # v1:
		# x = self.linear_relu_stack(x)
		# out = torch.concat([self.o_60(x), self.o_scalar(x)], dim=1)