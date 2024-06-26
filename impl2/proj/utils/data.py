in_vars_e = [
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
	("pbuf_CH4", 60), # 27-59 are constant (=)
	("pbuf_N2O", 60), # 27-59 are constant (=)
]

out_vars_e = [
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

zeroed_vars_e = [
	("ptend_q0001", 12), # 0-11 zeroed by submission weights
	("ptend_q0002", 15), # 0-14 zeroed by submission weights
	("ptend_q0003", 12), # 0-11 zeroed by submission weights
	("ptend_u", 12),     # 0-11 zeroed by submission weights
	("ptend_v", 12),     # 0-11 zeroed by submission weights
]

def expand_vars(vars: list[str]):
	return sum([[v[0]] if v[1] == 1 else [f"{v[0]}_{i}" for i in range(v[1])] for v in vars], start=[])

import json, numpy as np

# var names
in_vars = expand_vars(in_vars_e)
out_vars = expand_vars(out_vars_e)
in_len, out_len = len(in_vars), len(out_vars)
all_len = in_len + out_len
zeroed_vars = expand_vars(zeroed_vars_e)
non_zeroed_out_vars = [v for v in out_vars if v not in zeroed_vars]
all_vars = ["sample_id"] + in_vars + out_vars


# for plots
in_ticks = ([0, 60, 120, 180, 240, 300, 360, 376, 436, 496], ['state_t', 'state_q0001', 'state_q0002', 'state_q0003', 'state_u', 'state_v', 'cam', 'pbuf_ozone', 'pbuf_CH4', 'pbuf_N2O'])
out_ticks = ([0, 60, 120, 180, 240, 300, 360], ['ptend_t', 'ptend_q0001', 'ptend_q0002', 'ptend_q0003', 'ptend_u', 'ptend_v', 'cam_out'])

data_insights = json.load(open("data_insights.json"))
const_out_vars = [v for v in out_vars if data_insights[v]["std_dev"] == 0]

in_means = np.array([data_insights[v]["mean"] for v in in_vars])
out_means = np.array([data_insights[v]["mean"] for v in out_vars])
all_means = np.concatenate([in_means, out_means])

in_mins = np.array([data_insights[v]["min"] for v in in_vars])
out_mins = np.array([data_insights[v]["min"] for v in out_vars])
all_mins = np.concatenate([in_mins, out_mins])

in_maxs = np.array([data_insights[v]["max"] for v in in_vars])
out_maxs = np.array([data_insights[v]["max"] for v in out_vars])
all_maxs = np.concatenate([in_maxs, out_maxs])

in_mms = np.array([in_maxs[i] - in_mins[i] if in_maxs[i] - in_mins[i] != 0 else 1 for i in range(len(in_vars))])
out_mms = np.array([out_maxs[i] - out_mins[i] if out_maxs[i] - out_mins[i] != 0 else 1 for i in range(len(out_vars))])
all_mms = np.concatenate([in_mms, out_mms])

in_std_dev  = np.array([data_insights[v]["std_dev"] if data_insights[v]["std_dev"] != 0 else 1 for v in in_vars])
out_std_dev = np.array([data_insights[v]["std_dev"] if data_insights[v]["std_dev"] != 0 else 1 for v in out_vars])
all_std_dev = np.concatenate([in_std_dev, out_std_dev])

data_stats = {
	in_len: {
		"means": in_means,
		"mins": in_mins,
		"maxs": in_maxs,
		"mms": in_mms,
		"std_dev": in_std_dev,},
	out_len: {
		"means": out_means,
		"mins": out_mins,
		"maxs": out_maxs,
		"mms": out_mms,
		"std_dev": out_std_dev,},
	all_len: {
		"means": all_means,
		"mins": all_mins,
		"maxs": all_maxs,
		"mms": all_mms,
		"std_dev": all_std_dev,},
}

# if __name__ == "__main__":
# 	import matplotlib.pyplot as plt
# 	plt.errorbar(x=[i for i in range(all_len)], y=all_means, yerr=all_std_dev, uplims=True, lolims=True)
# 	plt.show()