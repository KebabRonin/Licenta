import fastkan, torch, utils.nn, utils.data
import kan
class MyKAN_parallel(torch.nn.Module):
	def __init__(self):
		super().__init__()
		self.kans = utils.nn.ParallelModuleList(torch.nn.ModuleList(
			fastkan.FasterKAN([utils.data.in_len, utils.data.in_len*2+1, 1], grid_range=[-100, 100], grid=10, device=utils.nn.device) for _ in range(utils.data.out_len)
		))
	def forward(self, x):
		return self.kans(x)
class MyKAN(torch.nn.Module):
	def __init__(self):
		super().__init__()
		self.kan = fastkan.FasterKAN([utils.data.in_len, utils.data.in_len*3+1, utils.data.out_len], grid_min=-100, grid_max=100, num_grids=10)
	def forward(self, x):
		return self.kan(x)