import gmr, cloudpickle
from my_utils import *
import polars as pl
from tqdm import trange

test = pl.read_parquet("Dataset/test/test.parquet")
sample_ids = test.drop_in_place("sample_id")
test.drop("sample_id")

weights = pl.read_csv("Dataset/weights.csv").drop('sample_id').cast(pl.Float64)
dlen = test.select(pl.len()).item()
submission = weights.clear(dlen)
out_schema=weights.schema
weights = weights.to_numpy()[0]
print("Read data")

batch_size = 2_500
model = cloudpickle.load(open("gmm_model.pickle", 'rb'))
submission = pl.concat([pl.DataFrame(model.predict(
		np.array([i for i in range(len(in_vars))]), test.slice(i, batch_size).to_numpy()
	), schema=out_schema) for i in trange(0, dlen, batch_size, desc='batches')])
submission.insert_column(0, sample_ids)
# submission = model.predict(np.array([i for i in range(test.shape[1])]), test.to_numpy())
submission.write_parquet("GMM/submission.parquet")

sys.stdout.reconfigure(encoding='utf-8')
print(submission)

del submission

a = pl.scan_parquet("GMM/submission.parquet")
assert len(a.columns) == len(out_vars) + 1, "Columns missing"
assert all([c in out_vars + ['sample_id'] for c in a.columns]), "Columns names are not OK"
assert a.select(pl.len()).collect().item() == 625_000, "Less than 625_000 rows found"