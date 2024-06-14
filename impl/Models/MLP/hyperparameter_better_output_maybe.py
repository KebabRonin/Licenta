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
from fastkan import FasterKAN
import cloudpickle, time

from model_def import MLP_60_split

DEVICE = torch.device("cuda")
DIR = os.getcwd()
EPOCHS = 6
TOTAL_SAMPLES = 10_091_520
N_TRAIN_SAMPLES = 50_000 / TOTAL_SAMPLES
N_VALID_SAMPLES = 10_000 / TOTAL_SAMPLES
PREPROC = 'minmax100'

def get_model_resnet(trial):
	torch.manual_seed(42)
	return MLP_60_split()

def identity2(x):
	return x #torch.tensor(x[0], dtype=dtype_mmm), torch.tensor(x[1], dtype=dtype_mmm)
def objective(trial):

	torch.set_default_dtype(torch.float64)
	MINIBATCH_SIZE = 4096
	floats  = trial.suggest_categorical("float", ['float64', 'float32', 'float16'])
	dtype_mmm = getattr(torch, floats)
	r2loss = R2Score(num_outputs=368).to(DEVICE)
	r2loss_img = R2Score(num_outputs=368, multioutput='raw_values').to(DEVICE)
	torch.set_default_dtype(dtype_mmm)
	model = get_model_resnet(trial).to(DEVICE)
	optimizer_name = "RAdam"
	loss_f = "MSELoss"
	loss_f = getattr(torch.nn, loss_f)()
	lr = 1e-4
	optimizer = getattr(optim, optimizer_name)(model.parameters(), lr=lr)
	for key, value in trial.params.items():
		print(f"	{key}: {value}")


	dset = CustomSQLDataset(norm_method=PREPROC)
	train_data, _, _ = tdata.random_split(dset, [N_TRAIN_SAMPLES, N_VALID_SAMPLES, 1 - N_TRAIN_SAMPLES - N_VALID_SAMPLES], generator=torch.Generator().manual_seed(50))
	_, valid_data, _ = tdata.random_split(CustomSQLDataset(norm_method='none'), [N_TRAIN_SAMPLES, N_VALID_SAMPLES, 1 - N_TRAIN_SAMPLES - N_VALID_SAMPLES], generator=torch.Generator().manual_seed(50))
	del _
	valid_loader = DataLoader(
						valid_data,
						batch_sampler=tdata.BatchSampler(tdata.SequentialSampler(valid_data), batch_size=50_000, drop_last=False),
						collate_fn=identity2,
						num_workers=1,
						persistent_workers=False,
						drop_last=False
					)

	# Training of the model.
	for epoch in tqdm.trange(EPOCHS, desc="Epochs", miniters=1, maxinterval=6000, position=0, leave=True):

		train_loader = DataLoader(
							train_data,
							batch_sampler=tdata.BatchSampler(tdata.RandomSampler(train_data, generator=torch.Generator().manual_seed(epoch)), batch_size=MINIBATCH_SIZE, drop_last=False), #
							collate_fn=identity2,
							num_workers=5,
							prefetch_factor=2,
							drop_last=False
						)
		model.train()
		for features, target in tqdm.tqdm(train_loader, desc="Batches", miniters=1, maxinterval=600, position=1, leave=True):
			features, target = torch.tensor(features, dtype=dtype_mmm), torch.tensor(target, dtype=dtype_mmm)
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
			pbar = tqdm.tqdm(valid_loader, desc="Valid Batches", miniters=1, maxinterval=600, position=2, leave=False)
			for features, target in pbar:
				features, target = preprocess_functions[PREPROC]['norm'](np.concatenate([features, target], axis=1))[0] , target
				features, target = torch.tensor(features, dtype=dtype_mmm), torch.tensor(target, dtype=torch.float64)
				features, target = features.to(DEVICE), target.to(DEVICE)
				output = model(features).cpu().numpy()
				output = torch.tensor(preprocess_functions[PREPROC]['denorm'](output), dtype=torch.float64, device=DEVICE)
				r2loss_total += r2loss(output, target).item()
				r2loss_img_rez = r2loss_img(output, target)
				r2loss_img_rez = r2loss_img_rez.to('cpu')
				cloudpickle.dump(r2loss_img_rez, open(f'{floats}_epoch_{epoch}_r2loss2.pkl', 'wb'))
				pbar.set_postfix_str(f'{r2loss_img_rez.mean():.4f}')
				batch_nr += 1

		accuracy = r2loss_total / batch_nr

		trial.report(accuracy, epoch)

		del features, target, output, r2loss_total
		torch.cuda.empty_cache()
		gc.collect()

		# Handle pruning based on the intermediate value.
		# if trial.should_prune():
		# 	raise optuna.exceptions.TrialPruned()

	torch.save(model, f"model_optuna_{time.strftime('%d-%b-%Y-%H-%M-%S')}_{floats}.pt", pickle_module=cloudpickle)
	del model, optimizer, loss_f
	torch.cuda.empty_cache()
	gc.collect()
	return accuracy


if __name__ == "__main__":
	study = optuna.create_study(study_name='resnet5-floats_fixed', direction="maximize", storage='postgresql://postgres:admin@localhost:5432/Data', load_if_exists=True, sampler=optuna.samplers.GridSampler({"float":['float64', 'float32', 'float16']}))
	study.optimize(objective, n_trials=3)

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