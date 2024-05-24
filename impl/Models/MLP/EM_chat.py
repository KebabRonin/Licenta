import numpy as np, polars as pl
from my_utils import *
import gmr, cloudpickle, time
from sklearn.mixture import GaussianMixture

n_samples = 50
train = pl.scan_parquet("Dataset/train/v1/train_1.parquet", n_rows=n_samples).drop('sample_id').cast(pl.Float32).collect()
print("Read",  time.strftime('%H:%M:%S', time.localtime()))

# Example data
train_in = normalize_subset(train, in_vars, method='none')
train_out = normalize_subset(train, out_vars, method='none')

# Combine X and Y into one array for fitting GMM
# train = np.hstack((train_in, train_out))

# Fit GMM
gmm = GaussianMixture(n_components=3, covariance_type='full', random_state=0, warm_start=True)
gmm.fit(train)
print("Fitted")
# Function to predict Y given X using the fitted GMM
def predict_gmm(gmm: GaussianMixture, X_new):
    n_features = X_new.shape[1]
    n_targets = gmm.means_.shape[1] - n_features

    # Extract GMM parameters
    means = gmm.means_
    covariances = gmm.covariances_

    # Separate means and covariances
    mu_X = means[:, :n_features]
    mu_Y = means[:, n_features:]
    sigma_XX = covariances[:, :n_features, :n_features]
    sigma_YY = covariances[:, n_features:, n_features:]
    sigma_XY = covariances[:, :n_features, n_features:]
    sigma_YX = covariances[:, n_features:, :n_features]

    # Initialize responsibilities for new data
    responsibilities = np.zeros((X_new.shape[0], gmm.n_components))

    for k in range(gmm.n_components):
        # Calculate the likelihood of the new X under each component
        diff = X_new - mu_X[k]
        sigma_XX_inv = np.linalg.inv(sigma_XX[k])
        exponent = -0.5 * np.einsum('ij,ij->i', diff @ sigma_XX_inv, diff)
        normalizer = np.sqrt((2 * np.pi) ** n_features * np.linalg.det(sigma_XX[k]))
        responsibilities[:, k] = np.exp(exponent) / normalizer * gmm.weights_[k]

    # Normalize responsibilities
    responsibilities /= responsibilities.sum(axis=1, keepdims=True)

    # Predict Y given X
    Y_pred = np.zeros((X_new.shape[0], n_targets))
    for k in range(gmm.n_components):
        # Conditional mean
        mu_Y_given_X = mu_Y[k] + sigma_YX[k] @ np.linalg.inv(sigma_XX[k]) @ (X_new - mu_X[k]).T
        Y_pred += responsibilities[:, k][:, np.newaxis] * mu_Y_given_X.T

    return Y_pred

# Predict for new data
valid = pl.scan_parquet("Dataset/train/v1/train_49.parquet", n_rows=50).drop('sample_id').cast(pl.Float32).collect()
valid_in = normalize_subset(valid, in_vars, method='none')
valid_out = normalize_subset(valid, out_vars, method='none')
print("Read2", time.strftime('%H:%M:%S', time.localtime()))
v_pred = predict_gmm(gmm, valid_in)
print(v_pred, time.strftime('%H:%M:%S', time.localtime()))
