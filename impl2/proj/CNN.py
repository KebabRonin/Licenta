import torch, torch.nn as nn
import numpy as np
from utils.nn import SkipConnection

class EncoderBlock(nn.Module):
	def __init__(self, in_channels, out_channels, act=nn.ReLU):
		super().__init__()
		self.model = nn.Sequential(
			nn.Conv1d(in_channels, out_channels, kernel_size=3, padding=1),
			# act(),
			nn.Conv1d(out_channels, out_channels, kernel_size=3, padding=1),
			# act(),
			nn.MaxPool1d(2, stride=2),
		)
	def forward(self, x):
		return self.model(x)

class DecoderBlock(nn.Module):
	def __init__(self, in_channels, out_channels):
		super().__init__()
		self.model = nn.Sequential(
			nn.ConvTranspose1d(in_channels, out_channels, kernel_size=2, stride=2),
			nn.Conv1d(out_channels, out_channels, kernel_size=3, padding=1),
			# act(),
			nn.Conv1d(out_channels, out_channels, kernel_size=3, padding=1),
			# act(),
		)
	def forward(self, x):
		return self.model(x)

class CNN(nn.Module):
	def __init__(self):
		super().__init__()
		self.model = nn.Sequential(
			EncoderBlock(10, 32),
			SkipConnection(nn.Sequential(
				EncoderBlock(32, 64),
				SkipConnection(nn.Sequential(
					EncoderBlock(64, 128),
					SkipConnection(nn.Sequential(
						EncoderBlock(128, 256),
						SkipConnection(nn.Sequential(
							EncoderBlock(256, 512),
							SkipConnection(
								nn.Sequential(
									torch.nn.Conv1d(512, 512, kernel_size=1),
									torch.nn.Conv1d(512, 512, kernel_size=1),
								), dim=1
							),
							DecoderBlock(1024, 256),
						), dim=2),
						DecoderBlock(256, 128),
					), dim=2),
					DecoderBlock(128, 64),
				), dim=2),
				DecoderBlock(64, 32),
			), dim=2),
			DecoderBlock(32, 7),
			nn.Flatten(),
			nn.Linear(7*256, 368),
		)

	def forward(self, x):
		padding = torch.zeros(x.shape[0], 60 - 16, device=x.device)
		x = torch.concatenate([x[:, :6*60], x[:, -3*60:], x[:, 6*60:-3*60], padding], dim=1)
		x = x.reshape(x.shape[0], 60, 10)
		x = torch.transpose(x, 1, 2)
		return self.model(x)

# torch.set_default_dtype(torch.float64)
# a = torch.tensor(np.arange(5*556, dtype=np.float64).reshape(5, 556))
# m = CNN()
# f = m(a)
# print(f.shape)
# print(a[:, 10], f[:, 10])
# print(a.shape, f.shape)