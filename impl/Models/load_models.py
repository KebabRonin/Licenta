import torch

models = ['MLP', 'XGB', 'CatBoost', 'CNN']

def load(model):
	match model:
		case 'MLP':
			return torch.load("")