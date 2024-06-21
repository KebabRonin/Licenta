from utils.preprocess import *
from utils.dset import *

if __name__ == "__main__":
	dd = get_splits(norm_method='none', dset_class=TimestepSQLDataset, fraction=0.05)[0]
	mock_in, mock_out = dd[:10]
	mock_all = np.concatenate([mock_in, mock_out], axis=1)
	print(mock_in.dtype)
	for k, v in preprocess_functions.items():
		erin, erout, erall = abs(mock_in - v['denorm'](v['norm'](mock_in))), abs(mock_out - v['denorm'](v['norm'](mock_out))), abs(mock_all - v['denorm'](v['norm'](mock_all)))
		print(f'\n=== {k} ===',
		f'Maximum precision loss: {erall.max()}; Mean precision loss: {erall.mean()}.'
		)