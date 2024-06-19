from utils.data import *

def preprocess_standardisation(arr:np.ndarray):
	stats = data_stats[arr.shape[1]]
	means, std_dev = stats["means"], stats["std_dev"]
	rez = ((arr - means) / std_dev )
	return rez
def preprocess_destandardisation(arr:np.ndarray):
	stats = data_stats[arr.shape[1]]
	means, std_dev = stats["means"], stats["std_dev"]
	rez  = ((arr * std_dev) + means)
	return rez

def preprocess_mean_normalisation(arr:np.ndarray):
	stats = data_stats[arr.shape[1]]
	mms, means = stats["mms"], stats["means"]
	features = ((arr - means ) / mms )
	return features
def preprocess_mean_denormalisation(arr:np.ndarray):
	stats = data_stats[arr.shape[1]]
	mms, means = stats["mms"], stats["means"]
	targets  = ((arr * mms) + means)
	return targets

def preprocess_standardisation_minmax(arr:np.ndarray, a, b):
	stats = data_stats[arr.shape[1]]
	mms, mins = stats["mms"], stats["mins"]
	features = a + (((arr - mins ) * (b-a)) / mms )
	return features
def preprocess_destandardisation_minmax(arr:np.ndarray, a, b):
	stats = data_stats[arr.shape[1]]
	mms, mins = stats["mms"], stats["mins"]
	targets  = (((arr - a) * mms) / (b-a) + mins)
	return targets

def preprocess_centered(arr:np.ndarray):
	stats = data_stats[arr.shape[1]]
	means = stats["means"]
	features = ((arr - means ))
	return features
def preprocess_decentered(arr:np.ndarray):
	stats = data_stats[arr.shape[1]]
	means = stats["means"]
	targets  = (arr + means)
	return targets

def preprocess_none(arr:np.ndarray):
	return arr
def preprocess_denone(arr:np.ndarray):
	return arr

# no closures or local objects so they can be pickled
def mmax10(x):
	return preprocess_standardisation_minmax(x, -10, 10)
def demmax10(x):
	return preprocess_destandardisation_minmax(x, -10, 10)
def mmax100(x):
	return preprocess_standardisation_minmax(x, -100, 100)
def demmax100(x):
	return preprocess_destandardisation_minmax(x, -100, 100)
def mmax01(x):
	return preprocess_standardisation_minmax(x, 0, 1)
def demmax01(x):
	return preprocess_destandardisation_minmax(x, 0, 1)

preprocess_functions = {
	"mean norm": {"norm": preprocess_mean_normalisation, "denorm": preprocess_mean_denormalisation},
	"minmax10": {"norm": mmax10, "denorm": demmax10},
	"minmax100": {"norm": mmax100, "denorm": demmax100},
	"minmax01": {"norm": mmax01, "denorm": demmax01},
	"standardisation": {"norm": preprocess_standardisation, "denorm": preprocess_destandardisation},
	"centered": {"norm": preprocess_centered, "denorm": preprocess_decentered},
	"none": {"norm": preprocess_none, "denorm": preprocess_denone},
}

if __name__ == "__main__":
	mock_len = 5_000
	mock_in, mock_out, mock_all = (np.random.rand(mock_len, in_len)), (np.random.rand(mock_len, out_len)), (np.random.rand(mock_len, all_len))
	print(mock_in.dtype)
	for k, v in preprocess_functions.items():
		print(f'\n=== {k} ===',
		'\n	MAE in :', abs(mock_in - v['denorm'](v['norm'](mock_in))).max(),
		'\n	MAE out:', abs(mock_out - v['denorm'](v['norm'](mock_out))).max(),
		'\n	MAE all:', abs(mock_all - v['denorm'](v['norm'](mock_all))).max(),
		)