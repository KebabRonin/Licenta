import torch.nn as nn
from utils.data import in_len, out_len
from utils.nn import *

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
class ResNet(nn.Module):
	def __init__(self, width=1024, act=nn.ReLU, outputs=out_len, depth=7):
		super().__init__()
		self.model = nn.Sequential(
				*[ResBlock(556 + width*i, width, act) for i in range(depth)],
				nn.Linear((556 + width*depth), outputs)
			)
	def forward(self, x):
		return self.model(x)

class ResNetParallel(nn.Module):
	def __init__(self, width=1024, act=nn.ReLU):
		super().__init__()
		self.model = ParallelModuleList(nn.ModuleList([
			ResNet(width, act, 60),
			ResNet(width, act, 60),
			ResNet(width, act, 60),
			ResNet(width, act, 60),
			ResNet(width, act, 60),
			ResNet(width, act, 60),
			ResNet(width, act, 8,depth=2),
		]))
	def forward(self, x):
		return self.model(x)

class ResNetParallelAll(nn.Module):
	def __init__(self, width=512, act=nn.ReLU):
		super().__init__()
		self.model = ParallelModuleList(nn.ModuleList([
			ResNet(width, act, 4, depth=3) for _ in range(int(368//4))
		]))
	def forward(self, x):
		return self.model(x)


if __name__ == '__main__':
	model = ResNet(1024, nn.SiLU)
	print(model)
	pytorch_total_params = sum(p.numel() for p in model.parameters())
	print(pytorch_total_params)
