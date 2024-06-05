import os

import optuna
from optuna.trial import TrialState
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as tdata
from torchmetrics.regression import R2Score
import tqdm, gc
from my_utils import *


DEVICE = torch.device("cuda")
DIR = os.getcwd()
EPOCHS = 10
TOTAL_SAMPLES = 10_091_520
N_TRAIN_SAMPLES = 150_000 / TOTAL_SAMPLES
N_VALID_SAMPLES = 10_000 / TOTAL_SAMPLES
N_TRIALS = 20

class ParallelModuleList(nn.Module):
	def __init__(self, models):
		super().__init__()

		# v3:
		self.models = models

	def forward(self, x):
		# v2:
		out = torch.concat([layer(x) for layer in (self.models)], dim=1)
		return out

def define_model_MLP_60_split(trial):
	# We optimize the number of layers, hidden units and dropout ratio in each layer.
	layers = []

	for j in range(6):
		dim_layers = []
		n_layers = trial.suggest_int(f"n_layers_{j}", 1, 5)
		in_features = 556
		for i in range(n_layers):
			out_features = trial.suggest_int(f"l{j}_{i}_n_units", 1, 4_096)
			l_act = trial.suggest_categorical(f"l{j}_{i}_act", ["ReLU", "LeakyReLU", "ELU", "Tanh", "SiLU"])
			dim_layers.append(nn.Linear(in_features, out_features))
			dim_layers.append(nn.LayerNorm(out_features))
			dim_layers.append(getattr(torch.nn, l_act)())
			in_features = out_features
		dim_layers.append(nn.Linear(in_features, 60))
		layers.append(nn.Sequential(*dim_layers))
	dim_layers = []
	n_layers = trial.suggest_int(f"n_layers_last8", 1, 3)
	in_features = 556
	for i in range(n_layers):
		out_features = trial.suggest_int(f"llast8_{i}_n_units", 1, 2_048)
		l_act = trial.suggest_categorical(f"llast8_{i}_act", ["ReLU", "LeakyReLU", "ELU", "Tanh", "SiLU"])
		dim_layers.append(nn.Linear(in_features, out_features))
		dim_layers.append(nn.LayerNorm(out_features))
		dim_layers.append(getattr(torch.nn, l_act)())
		in_features = out_features
	dim_layers.append(nn.Linear(in_features, 8))
	layers.append(nn.Sequential(*dim_layers))

	return ParallelModuleList(nn.ModuleList(layers))

def define_model_MLP_simple(trial):
	# We optimize the number of layers, hidden units and dropout ratio in each layer.
	n_layers = trial.suggest_int("n_layers", 1, 7)
	layers = []

	in_features = 556
	for i in range(n_layers):
		out_features = trial.suggest_int(f"l{i}_n_units", 1, 4_096)
		l_act = trial.suggest_categorical(f"l{i}_act", ["ReLU", "LeakyReLU", "ELU", "Tanh", "SiLU"])
		layers.append(nn.Linear(in_features, out_features))
		layers.append(nn.LayerNorm(out_features))
		layers.append(getattr(torch.nn, l_act)())

		in_features = out_features

	return nn.Sequential(SkipConnection(nn.Sequential(*layers)), nn.Linear(556 + in_features, 368))

def identity(x: tuple[np.ndarray, np.ndarray]):
	return torch.tensor(x[0], dtype=torch.float64), torch.tensor(x[1], dtype=torch.float64)

def objective(trial):
	# Generate the model.
	if trial.suggest_categorical("model_type", ['60_split', 'simple']) == '60_split':
		model = define_model_MLP_60_split(trial).to(DEVICE)
	else:
		model = define_model_MLP_simple(trial).to(DEVICE)
	# Generate the optimizers.
	optimizer_name = trial.suggest_categorical("optimizer", ["RAdam", "Adadelta", "AdamW", "Adam", "NAdam", "Adamax"])
	loss_f = trial.suggest_categorical("loss", ["L1Loss", "MSELoss", "SmoothL1Loss"])
	loss_f = getattr(torch.nn, loss_f)()
	lr = trial.suggest_float("lr", 1e-5, 1e-1, log=True)
	optimizer = getattr(optim, optimizer_name)(model.parameters(), lr=lr)

	MINIBATCH_SIZE = trial.suggest_int("mini_batch_size", 1, 5_000)
	NORMALIZATION  = trial.suggest_categorical("normalization", preprocess_functions.keys())
	# print(f"{NORMALIZATION=} | {MINIBATCH_SIZE=}")
	for key, value in trial.params.items():
		print(f"	{key}: {value}")

	dset = CustomSQLDataset(norm_method=NORMALIZATION)
	r2loss = R2Score(num_outputs=368).to(DEVICE)
	train_data, valid_data, _ = tdata.random_split(dset, [N_TRAIN_SAMPLES, N_VALID_SAMPLES, 1 - N_TRAIN_SAMPLES - N_VALID_SAMPLES], generator=torch.Generator().manual_seed(50))
	del _
	valid_loader = DataLoader(
						valid_data,
						batch_sampler=tdata.BatchSampler(tdata.SequentialSampler(valid_data), batch_size=10_000, drop_last=False),
						collate_fn=identity,
						num_workers=1,
						persistent_workers=False,
						drop_last=False
						# pin_memory=True, # I already send the tensors to cuda, so this might be useless
						# pin_memory_device='cuda',
						# prefetch_factor=4,
					)

	# Training of the model.
	for epoch in tqdm.trange(EPOCHS, desc="Epochs", miniters=1, maxinterval=6000, position=0, leave=True):

		train_loader = DataLoader(
							train_data,
							batch_sampler=tdata.BatchSampler(tdata.RandomSampler(train_data, generator=torch.Generator().manual_seed(epoch)), batch_size=MINIBATCH_SIZE, drop_last=False), #
							collate_fn=identity,
							num_workers=5,
							# persistent_workers=True,
							# pin_memory=True,
							# pin_memory_device='cuda',
							prefetch_factor=2,
							drop_last=False
						)
		model.train()
		for features, target in tqdm.tqdm(train_loader, desc="Batches", miniters=1, maxinterval=600, position=1, leave=True):
			features, target = features.to(DEVICE), target.to(DEVICE)
			optimizer.zero_grad()
			output = model(features)
			loss = loss_f(output, target)
			loss.backward()
			optimizer.step()
			del features, target, loss, output
			torch.cuda.empty_cache()
			gc.collect()
		del train_loader

		# Validation of the model.
		model.eval()
		r2loss_total = 0
		with torch.no_grad():
			batch_nr = 0
			for features, target in tqdm.tqdm(valid_loader, desc="Valid Batches", miniters=1, maxinterval=600, position=2, leave=False):
				features, target = features.to(DEVICE), target.to(DEVICE)
				output = model(features)
				r2loss_total += r2loss(output, target).item()
				batch_nr += 1

		accuracy = r2loss_total / batch_nr

		trial.report(accuracy, epoch)

		del features, target, output, r2loss_total
		torch.cuda.empty_cache()
		gc.collect()

		# Handle pruning based on the intermediate value.
		if trial.should_prune():
			raise optuna.exceptions.TrialPruned()

	del model, optimizer, loss_f
	torch.cuda.empty_cache()
	gc.collect()
	return accuracy


if __name__ == "__main__":
	study = optuna.create_study(study_name='MLP_renewed2', direction="maximize", storage='postgresql://postgres:admin@localhost:5432/Data', load_if_exists=True)
	study.optimize(objective, n_trials=N_TRIALS)

	pruned_trials = study.get_trials(deepcopy=False, states=[TrialState.PRUNED])
	complete_trials = study.get_trials(deepcopy=False, states=[TrialState.COMPLETE])

	print("Study statistics: ")
	print("  Number of finished trials: ", len(study.trials))
	print("  Number of pruned trials: ", len(pruned_trials))
	print("  Number of complete trials: ", len(complete_trials))
	print("Best params: ", study.best_params)
	print("Best value: ", study.best_value)
	print("Best Trial: ", study.best_trial)
	print("Trials: ", study.trials)
	trial = study.best_trial
	print("  Value: ", trial.value)
	print("  Params: ")
	for key, value in trial.params.items():
		print("	{}: {}".format(key, value))