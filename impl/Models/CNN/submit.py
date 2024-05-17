import polars as pl, torch #, numpy as np, pandas as pd
import sys, time, json, asyncio
from model_def import model
# from my_utils import in_vars, out_vars, zeroed_vars
from tqdm import trange
from sklearn.metrics import r2_score
from model_def import model
sys.stdout.reconfigure(encoding='utf-8')


# def replace_row(self:pl.DataFrame, row_index: str, new_values):
#     return self.select(
#     (pl.when(pl.col('sample_id').eq(row_index))
#         .then(pl.lit(x))
#         .otherwise(pl.col(self.columns[1+i])))
#         .alias(self.columns[1+i]) for i, x in enumerate(new_values)
# )
# pl.DataFrame.replace_row=replace_row
# del replace_row

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
test = pl.read_parquet("Dataset/test/test.parquet")
weights = pl.read_csv("Dataset/submission_weights.csv")
dlen = test.select(pl.len()).item()
submission = weights.clear(dlen)
submission.replace_column(0, test['sample_id'])
weights = torch.tensor(weights.drop("sample_id").to_numpy()[0], device=device) # [0].to(device)

def predict(name):
	sample = test.row(by_predicate=pl.col('sample_id').eq(name[0]))
	prediction = model(torch.tensor(list(sample[1:])).to(device))
	prediction = (prediction * weights).to('cpu')
	weighted = tuple(([sample[0]] + prediction.tolist()))
	return weighted
async def get_pred(start, end):
	sample = test.slice(start, end)
	prediction = model(torch.tensor(sample[1:].to_numpy(), device=device))
	weighted = pl.DataFrame((prediction * weights).to('cpu').numpy())
	df = weighted.with_columns(sample_id=sample.select(pl.col('sample_id')))
	return df

batch_size = 50_000

with torch.no_grad():
	model = torch.load('model.pt')
	submission = submission.map_rows(predict)

	submission = pl.concat(asyncio.gather([get_pred(i, batch_size) for i in trange(0, dlen, batch_size)]))

	# submission = submission.with_columns(test['sample_id'])
	# for sample in tqdm(test.iter_rows(), total=dlen, leave=False, miniters=500):
	# 	prediction = model(torch.tensor(sample[1:]).to('cuda')).to('cpu')
	# 	prediction = (prediction.numpy() * weights)[0]
	# 	weighted = pl.DataFrame([sample[0]] + prediction.tolist(), schema=submission.schema)
	# 	print(weighted)
	# 	submission = pl.concat([submission, weighted])
	# 	print(submission)
	# 	break

	submission.write_csv('submission.csv')
