import matplotlib.pyplot as plt
import dill, utils.preprocess as pp, utils.data as ud, numpy as np

def plot_values():
	sdevh, sdevlo = pp.preprocess_standardisation(ud.out_std_dev[np.newaxis, :]+ud.out_means[np.newaxis, :]).squeeze(), pp.preprocess_standardisation(-ud.out_std_dev[np.newaxis, :]+ud.out_means[np.newaxis, :]).squeeze()
	mn, mx = pp.preprocess_standardisation(ud.out_mins[np.newaxis, :]).squeeze(), pp.preprocess_standardisation(ud.out_maxs[np.newaxis, :]).squeeze()
	# plt.plot(sdev.squeeze(), label='sdev')
	plt.fill_between([i for i in range(368)], sdevh, sdevlo, color='red', alpha=0.2, label='std_dev')
	# plt.plot([0] * 368, label='mean', color='black')
	plt.plot(mn.squeeze(), label='min')
	plt.plot(mx.squeeze(), label='max')
	# plt.fill_between([i for i in range(368)], mn.squeeze(), mx.squeeze(), color='blue', alpha=0.2, label='value range')
	plt.legend()
	plt.yscale('symlog')
	plt.xticks(*ud.out_ticks)
	plt.axis((0, 367, None, None))
	plt.grid()
	plt.show()

def plot_model(model_path):
	d = dill.load(open(model_path, 'rb'), ignore=True)
	# print(d)

	intervals = list(map(len, d['train_losses']))
	prev = 0
	for i in range(len(intervals)):
		intervals[i] += prev
		prev += intervals[i]
	if len(d['losses']) != len(intervals):
		intervals = [0] + intervals
	plt.plot(d['losses'])
	plt.show()
	# plt.plot(intervals, d['losses'])
	# # for x in d['train_losses']:
	# # 	plt.plot(x)
	# plt.plot(sum(d['train_losses'], []))
	# plt.show()
import dill
dill.load(open('model.pickle', 'rb'))
# plot_model(r'Models\kan\model_checkpoint_batch_4_epoch_29.pickle')