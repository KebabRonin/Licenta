import numpy as np
import torch, polars as pl
import torch.nn as nn
import torch.nn.functional as F
from my_utils import *

in_len = 556
out_len = 368
#  ['model_type': 60_split, 'n_layers_0': 3, 'l0_0_n_units': 1387, 'l0_0_act': Tanh, 'l0_1_n_units': 2947, 'l0_1_act': LeakyReLU, 'l0_2_n_units': 2563, 'l0_2_act': LeakyReLU, 'n_layers_1': 2, 'l1_0_n_units': 29, 'l1_0_act': Tanh, 'l1_1_n_units': 3078, 'l1_1_act': SiLU, 'n_layers_2': 3, 'l2_0_n_units': 3195, 'l2_0_act': ReLU, 'l2_1_n_units': 2672, 'l2_1_act': ReLU, 'l2_2_n_units': 1930, 'l2_2_act': Tanh, 'n_layers_3': 3, 'l3_0_n_units': 2602, 'l3_0_act': SiLU, 'l3_1_n_units': 2974, 'l3_1_act': SiLU, 'l3_2_n_units': 1180, 'l3_2_act': Tanh, 'n_layers_4': 1, 'l4_0_n_units': 3071, 'l4_0_act': ReLU, 'n_layers_5': 4, 'l5_0_n_units': 2923, 'l5_0_act': LeakyReLU, 'l5_1_n_units': 4, 'l5_1_act': ELU, 'l5_2_n_units': 1321, 'l5_2_act': 'LeakyReLU', 'l5_3_n_units': 4059, 'l5_3_act': ReLU, 'n_layers_last8': 1, 'llast8_0_n_units': 69, 'llast8_0_act': Tanh, 'optimizer': Adadelta, 'loss': 'MSELoss', 'lr': 0.014666056303807039, 'mini_batch_size': 3796, 'normalization': none]
l_widths = [556, 1024, 512, 368]
class ResBlock(nn.Module):
	def __init__(self, in_features, width=1024, act=nn.ReLU):
		super().__init__()
		self.model = SkipConnection(nn.Sequential(
			nn.Linear(in_features, width),
			nn.LayerNorm(width),
			act(),
			nn.Linear(width, width),
			nn.LayerNorm(width),
			act(),
		))
	def forward(self, x):
		return self.model(x)
class MLP_60_split(nn.Module):
	def __init__(self, width=1024, act=nn.ReLU):
		super().__init__()
		self.model = ParallelModuleList(nn.ModuleList([
			nn.Sequential(
				ResBlock(556, width, act),
				ResBlock(556 + width*1, width, act),
				ResBlock(556 + width*2, width, act),
				ResBlock(556 + width*3, width, act),
				ResBlock(556 + width*4, width, act),
				ResBlock(556 + width*5, width, act),
				nn.Linear((556 + width*6), 60)
			) for _ in range (6)
		] + [
			nn.Sequential(
				ResBlock(556, width, act),
				nn.Linear(556 + width, 8)
			)
		]))
		"""# self.model = ParallelModuleList(nn.ModuleList([
		# 	nn.Sequential(
		# 		SkipConnection(nn.Sequential(
		# 			nn.Linear(556, 1387),
		# 			nn.LayerNorm(1387),
		# 			nn.Tanh(),
		# 			nn.Linear(1387, 2947),
		# 			nn.LayerNorm(2947),
		# 			nn.LeakyReLU(),
		# 		)),
		# 		nn.Linear(556 + 2947, 2563),
		# 		nn.LayerNorm(2563),
		# 		nn.LeakyReLU(),
		# 		nn.Linear(2563, 60)
		# 	), # 0
		# 	nn.Sequential(
		# 		SkipConnection(nn.Sequential(
		# 			nn.Linear(556, 300),
		# 			nn.LayerNorm(300),
		# 			nn.Tanh(),
		# 		)),
		# 		nn.Linear(556 + 300, 3078),
		# 		nn.LayerNorm(3078),
		# 		nn.SiLU(),
		# 		nn.Linear(3078, 60)
		# 	), # 1
		# 	nn.Sequential(
		# 		SkipConnection(nn.Sequential(
		# 			nn.Linear(556, 3195),
		# 			nn.LayerNorm(3195),
		# 			nn.ReLU(),
		# 			nn.Linear(3195, 2672),
		# 			nn.LayerNorm(2672),
		# 			nn.ReLU(),
		# 			nn.Linear(2672, 1930),
		# 			nn.LayerNorm(1930),
		# 			nn.Tanh(),
		# 		)),
		# 		nn.Linear(556 + 1930, 1930),
		# 		nn.LayerNorm(1930),
		# 		nn.ReLU(),
		# 		nn.Linear(1930, 60)
		# 	), # 2
		# 	nn.Sequential(
		# 		SkipConnection(nn.Sequential(
		# 			nn.Linear(556, 2602),
		# 			nn.LayerNorm(2602),
		# 			nn.SiLU(),
		# 			nn.Linear(2602, 2974),
		# 			nn.LayerNorm(2974),
		# 			nn.SiLU(),
		# 		)),
		# 		SkipConnection(nn.Sequential(
		# 			nn.Linear(556 + 2974, 1180),
		# 			nn.LayerNorm(1180),
		# 			nn.Tanh(),
		# 		)),
		# 		nn.Linear(556 + 2974 + 1180, 60)
		# 	), # 3
		# 	nn.Sequential(
		# 		SkipConnection(nn.Sequential(
		# 			nn.Linear(556, 3071),
		# 			nn.LayerNorm(3071),
		# 			nn.ReLU(),
		# 		)),
		# 		nn.Linear(556 + 3071, 60)
		# 	), # 4
		# 	nn.Sequential(
		# 		nn.Linear(556, 2923),
		# 		nn.LayerNorm(2923),
		# 		nn.LeakyReLU(),
		# 		nn.Linear(2923, 400),
		# 		nn.LayerNorm(400),
		# 		nn.ELU(),
		# 		nn.Linear(400, 1321),
		# 		nn.LayerNorm(1321),
		# 		nn.LeakyReLU(),
		# 		nn.Linear(1321, 4059),
		# 		nn.LayerNorm(4059),
		# 		nn.SiLU(),
		# 		nn.Linear(4059, 60)
		# 	), # 5
		# 	nn.Sequential(
		# 		SkipConnection(nn.Sequential(
		# 			nn.Linear(556, 69),
		# 			nn.LayerNorm(69),
		# 			nn.Tanh(),
		# 		)),
		# 		nn.Linear(556 + 69, 8)
		# 	), # last_8
		# ]))"""

	def forward(self, x):
		return self.model(x)

class MLP(nn.Module):
	def forward(self, x):
		return self.model(x)

model = MLP_60_split(1024, nn.SiLU)

if __name__ == '__main__':
	print(model)
	pytorch_total_params = sum(p.numel() for p in model.parameters())
	print(pytorch_total_params)
