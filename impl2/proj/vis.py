import matplotlib.pyplot as plt
import dill

# for i in range(4):
d = dill.load(open(r'Models\resnet_parallel\model_checkpoint_batch_4_epoch_4.pickle', 'rb'), ignore=True)
print(d)

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
plt.plot(sum(d['train_losses'], []))
plt.show()