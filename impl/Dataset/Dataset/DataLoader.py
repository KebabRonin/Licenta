in_vars = [
	("state_t", 60),
	("state_q0001", 60),
	("state_q0002", 60),
	("state_q0003", 60),
	("state_u", 60),
	("state_v", 60),
	("state_ps", 1),
	("pbuf_SOLIN", 1),
	("pbuf_LHFLX", 1),
	("pbuf_SHFLX", 1),
	("pbuf_TAUX", 1),
	("pbuf_TAUY", 1),
	("pbuf_COSZRS", 1),
	("cam_in_ALDIF", 1),
	("cam_in_ALDIR", 1),
	("cam_in_ASDIF", 1),
	("cam_in_ASDIR", 1),
	("cam_in_LWUP", 1),
	("cam_in_ICEFRAC", 1),
	("cam_in_LANDFRAC", 1),
	("cam_in_OCNFRAC", 1),
	("cam_in_SNOWHLAND", 1),
	("pbuf_ozone", 60),
	("pbuf_CH4", 60), # 27-59 dropped because constant (=)
	("pbuf_N2O", 60), # 27-59 dropped because constant (=)
]

out_vars = [
	("ptend_t", 60),
	("ptend_q0001", 60), # 0-11 zeroed by submission weights
	("ptend_q0002", 60), # 0-14 zeroed by submission weights
	("ptend_q0003", 60), # 0-11 zeroed by submission weights
	("ptend_u", 60),     # 0-11 zeroed by submission weights
	("ptend_v", 60),     # 0-11 zeroed by submission weights
	("cam_out_NETSW", 1),
	("cam_out_FLWDS", 1),
	("cam_out_PRECSC", 1),
	("cam_out_PRECC", 1),
	("cam_out_SOLS", 1),
	("cam_out_SOLL", 1),
	("cam_out_SOLSD", 1),
	("cam_out_SOLLD", 1),
]

def expand_vars(vars: list[str]):
	return sum([[v[0]] if v[1] == 1 else [f"{v[0]}_{i}" for i in range(v[1])] for v in vars], start=[])

vars = in_vars + out_vars
actual_vars = expand_vars(vars)

import pandas as pd
import polars as pl
import matplotlib.pyplot as plt
import sys
sys.stdout.reconfigure(encoding='utf-8')

def figs():
	insights = pl.read_csv("Dataset/Dataset/data_insights.csv")
	dset = pl.scan_parquet(f"Dataset/Dataset/train/v2/*.parquet")

	bins = 25
	vvs = []
	xxs = []
	yys = []

	for idx, var in enumerate(insights.iter_rows()):
		print(var)
		# start = dset.select(pl.col(var[0]).min()).collect()[var[0]][0]
		# end = dset.select(pl.col(var[0]).max()).collect()[var[0]][0]
		# step = (end - start)/(bins)
		# xs = [start + step*i for i in range(bins+1)]
		step = (var[2]*10/bins)
		xs = [var[1] + step*i for i in range(-bins//2, bins//2+bins%2)]
		# print(xs)
		before = [dset.select(pl.col(var[0]).lt(xs[0]).sum()).collect()[var[0]][0]]
		after = [dset.select(pl.col(var[0]).gt(xs[-1]).sum()).collect()[var[0]][0]]
		ys = [dset.select(pl.col(var[0]).is_between(xs[i-1], xs[i]).sum()).collect()[var[0]][0] for i in range(1, len(xs))]

		ys = before + ys + after
		xs = xs + [xs[-1]+step]
		# xs = xs[:-1]
		# print(xs, ys)
		plt.clf()
		bb = plt.bar(xs, ys, color='blue')
		bb[0].set_color('orange')
		bb[-1].set_color('orange')
		plt.title(var[0])
		plt.savefig(f"figs/{var[0]}.png")
		vars.append(var[0])
		xxs.append(xs)
		yys.append(ys)
		# plt.show()
		# a["var_name"][idx] = var
		# a["xs"][idx] = xs
		# a["ys"][idx] = ys
	# dset.plot.hist(actual_vars[1], )
	# for i in range(25):
	# 	dset = pl.scan_parquet(f"Dataset/polars/train_batch/train_{i}.parquet")

	a = pl.DataFrame({"var_name": vars, "xs": xxs, "ys": yys}, schema={"var_name": pl.String, "xs": pl.Array(pl.Float32, 51), "ys": pl.Array(pl.Float32, 51)})
	a.write_parquet("hists")

def avgs():
	dset = pl.scan_parquet(f"Dataset/train/v2/*.parquet")
	d = dict()
	ctx = pl.SQLContext(population=dset, eager_execution=True)
	query = "SELECT AVG({cname}) as avg, STDDEV({cname}) as stddev FROM population"
	for cname, size in vars:
		if size > 1:
			for s in range(60):
				ccname = f"{cname}_{s}"
				if ccname in dset.columns:
					d[ccname] = ctx.execute(query.format(cname=ccname))
					d[ccname] = {"mean":d[ccname]['avg'][0], "stddev":d[ccname]['stddev'][0]}
					print(f"{ccname:<20} - {d[ccname]}")
		else:
			d[cname] = ctx.execute(query.format(cname=cname))
			d[cname] = {"mean":d[cname]['avg'][0], "stddev":d[cname]['stddev'][0]}
			print(f"{cname:<20} - {d[cname]}")
	print(d)
	with open("data_insights", "wt") as f:
		f.write(d)

# dset = pl.scan_parquet(f"Dataset/train/v2/*.parquet")
# # print(dset.columns)
# ctx = pl.SQLContext(population=dset, eager_execution=True)
# query = "SELECT AVG({cname}) as avg, STDDEV({cname}) as stddev FROM population"
# d = ctx.execute(query.format(cname=actual_vars[1]))
# print(d)
# print({"mean":d[actual_vars[1]]['avg'][0], "stddev":d[actual_vars[1]]['stddev'][0]})
# zeroed = []
# weights = pd.read_csv("Dataset/submission_weights.csv")
# print(len(weights.columns))
# for i in weights.columns:
# 	if weights[i][0] == 0:
# 		zeroed.append(i)


# print(zeroed)