# import bigframes.dataframe
import google
import google.auth
from google.cloud import aiplatform, bigquery
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

client = bigquery.Client(credentials=creds[0])
query = "SELECT sample_id, cam_out_SOLLD, cam_out_SOLSD, ML.MIN_MAX_SCALER(cam_out_SOLLD) over(), ML.MIN_MAX_SCALER(cam_out_SOLSD) over() from `licenta-426817.train_climsim.train` limit 100" # WHERE mod(cast(substring(sample_id, 7) as int64), 384) = 10
# dataset = client.dataset('climsim-dataset')
# table = dataset.table('train')
# import bigframes
# bigframes.dataframe.DataFrame()
# import optuna
import coiled, dask, optuna
# optuna.create_study()
results = client.query(query)
for i in results:
	print(i)
# print(results)