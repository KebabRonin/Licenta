import torch, torch.nn as nn

class CNN(nn.Module):
	def __init__(self):
		super().__init__()
		self.model = nn.Sequential(
			nn.Conv1d(1, 64, kernel_size=3, padding=1),
			nn.LeakyReLU(),
			nn.Conv1d(64, 64, kernel_size=3, padding=1),
			nn.LeakyReLU(),
			nn.MaxPool2d(2, stride=2),
		)