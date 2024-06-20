# import bigframes.dataframe
import dask.distributed
import google
import utils.data
import google.auth
from google.cloud import aiplatform, bigquery
import optuna, sys
sys.stdout.reconfigure(encoding='utf-8')
if __name__ == '__main__':
	creds = google.auth.load_credentials_from_file('g-cloud-secret.json')
	print(creds)
	aiplatform.init(
		project='Licenta',
		# experiment='experiment',
		# staging_bucket=staging_bucket,
		credentials=creds[0],
		# encryption_spec_key_name=encryption_spec_key_name,
		service_account='admin-758@licenta-426817.iam.gserviceaccount.com',
	)
	import numpy as np
	client = bigquery.Client(credentials=creds[0])
	import time
	start = time.time()
	arrs = ','.join([f'"train_{i*10}"' for i in range(5_000)])
	query = f"SELECT * from `licenta-426817.train_climsim.train` WHERE `sample_id` in ({arrs})"
	# dataset = client.dataset('climsim-dataset')
	# table = dataset.table('train')
	# from bigframes.dataframe import DataFrame
	# import optuna
	# optuna.create_study()
	results = client.query(query)
	a = results.to_dataframe()
	print(time.time() - start)
	print(a)
	# for i in a:
	# 	print(i)

	# print(results)