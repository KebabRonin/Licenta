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
		# print(x.shape)
		return torch.concat([self.actual(x), x], dim=1)

l_widths = [768, 640, 512, 640, 640, 360]
d1 = 2048
d2 = 1024
d3 = 512

class MLP(nn.Module):
	def __init__(self):
		super().__init__()

		# v3:
		self.modelss = nn.ModuleList([
			*(nn.Sequential(
				ResBlock(nn.Sequential(
					nn.Linear(in_len, d2),
					nn.LeakyReLU(),
					nn.Linear(d2, d3),
					nn.LeakyReLU(),
					nn.Linear(d3, 256),
					nn.LeakyReLU(),
				)),
				nn.Linear(256+in_len, 20),
			) for _ in range(360//20)),
			nn.Sequential(
				ResBlock(nn.Sequential(
					nn.Linear(in_len, d2),
					nn.LeakyReLU(),
					nn.Linear(d2, d3),
					nn.LeakyReLU(),
					nn.Linear(d3, 256),
					nn.LeakyReLU(),
				)),
				nn.Linear(256+in_len, 8),
			)
		])

	def forward(self, x):
		# v2:
		out = torch.concat([layer(x) for layer in (self.modelss)], dim=1)
		return out

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