import torch.nn as nn
from utils.data import in_len, out_len, all_len
from utils.nn import *

class MLP(nn.Module):
	def __init__(self, hidden_sizes, output_size):
		super().__init__()

		layers = []
		previous_size = in_len
		for hidden_size in hidden_sizes:
			layers.append(nn.Linear(previous_size, hidden_size))
			layers.append(nn.LayerNorm(hidden_size))
			layers.append(nn.LeakyReLU(inplace=True))
			layers.append(nn.Dropout(p=0.1))
			previous_size = hidden_size

		layers.append(nn.Linear(previous_size, output_size))

		self.layers = nn.Sequential(*layers)
	def forward(self, x):
		return self.layers(x)


if __name__ == '__main__':
	model = MLP([3*all_len, 2*all_len, 2*all_len, 2*all_len, 3*all_len], out_len)
	optimizer= torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)
	loss = torch.nn.MSELoss()
	batch_size = 20
	model_name = 'mlp_simple_bottleneck'
