import numpy as np
import torch, polars as pl
import torch.nn as nn
import torch.nn.functional as F
from my_utils import *

in_len = 556
out_len = 368

from kan import KAN
torch.set_default_dtype(torch.float64)
model = KAN([in_len, in_len + out_len, out_len], grid=5, grid_range=[-100, 100], device=DEVICE)

if __name__ == '__main__':
	print(model)
	model(torch.tensor(np.arange(in_len).reshape(1, in_len), device=DEVICE))
	model.plot()
	pytorch_total_params = sum(p.numel() for p in model.parameters())
	print(pytorch_total_params)
