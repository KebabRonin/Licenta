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

batch_size = 6_250
model = cloudpickle.load(open("gmm_model.pickle", 'rb'))
submission = pl.concat([pl.DataFrame(model.predict(
		np.array([i for i in range(len(in_vars))]), test.slice(i, batch_size).to_numpy()
	), schema=out_schema) for i in trange(0, dlen, batch_size)])
submission.insert_column(0, sample_ids)
print(submission)
# submission = model.predict(np.array([i for i in range(test.shape[1])]), test.to_numpy())
submission.write_parquet("GMM/submission.parquet")