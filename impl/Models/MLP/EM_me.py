import numpy as np, polars as pl
from my_utils import *
import cloudpickle, time


class EMRegressor:
	def __init__(self, n_components, data_len=(556+368)):
		self.n_components = n_components
		self.mean = np.random.rand(n_components)
		self.std  = np.random.rand(n_components)
		self.data_len = data_len
		self.weights = np.full(shape=(data_len, self.n_components), fill_value=1/self.n_components, dtype=np.float32)
		self.predictions = None
		self.z = None

	def Estep(self):
		mulw = (self.predictions[None, :, :] * self.weights).squeeze(axis=0)
		other = mulw[:, :, None].squeeze(axis=2) # prepare for elementwise division
		fp = self.final_pred[:, :, np.newaxis]   # prepare for elementwise division
		print(f"{mulw.shape=}, {other.shape=}, {fp.shape=}")
		self.z = other / fp # dout[:, :, np.newaxis]
		print(f"EStep {self.z.shape=}")
		print(f"{self.z=}")

	def Mstep(self):
		print(f"{self.weights[0]=}")
		self.weights = np.mean(self.z, axis=0)
		print(f"MStep {self.weights.shape=}")
		print(f"{self.weights[0]=}")

	def train(self, data_in: np.ndarray, data_out: np.ndarray, n_iters=10):
		if data_in.shape[1] != self.in_len:
			raise Exception(f"Input data shape ({data_in.shape}) doesn't fit with the specified input feature length ({self.in_len})")
		# self.predict(data_in) # updates self.prediction matrix (samples x output features x models)
		for i in range(n_iters):
			self.predict(data_in) # updates self.prediction matrix (samples x output features x models)
			self.Estep(data_out)
			self.Mstep()
			print(f"Iter {i+1:>3} weights:\n{self.weights[:20, :].T}")
