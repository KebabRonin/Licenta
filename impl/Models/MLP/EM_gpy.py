import numpy as np, polars as pl
from my_utils import *
from tqdm import trange, tqdm
import gmr, cloudpickle, time, gc
# from sklearn.mixture import GaussianMixture
from torchmetrics.regression import R2Score
import gpytorch

def train(model, train_x, train_y, n_iter=10):
    """Train the model.

    Arguments
    model   --  The model to train.
    train_x --  The training inputs.
    train_y --  The training labels.
    n_iter  --  The number of iterations.
    """
    model.train()
    optimizer = torch.optim.LBFGS(model.parameters(), line_search_fn='strong_wolfe')
    likelihood = model.likelihood
    mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)
    def closure():
        optimizer.zero_grad()
        output = model(train_x)
        loss = -mll(output, train_y)
        loss.backward()
        return loss
    for i in range(n_iter):
        loss = optimizer.step(closure)
        if (i + 1) % 1 == 0:
            print(f'Iter {i + 1:3d}/{n_iter} - Loss: {loss.item():.3f}')
    model.eval()

dset = CustomSQLDataset(norm_method="none")
splits = get_splits()
# print(splits)
trs = tdata.random_split(dset, splits, generator=torch.Generator().manual_seed(50))
batch = trs[0]
sqldloader = DataLoader(batch, num_workers=0,
							batch_sampler=tdata.BatchSampler(tdata.RandomSampler(batch), batch_size=5, drop_last=False), #
							collate_fn=identity)
train_x, train_y = next(iter(sqldloader))

# initialize likelihood and model
likelihood = gpytorch.likelihoods.GaussianLikelihood()
class ExactGP(gpytorch.models.ExactGP):
    """Exact Gaussian Process model.

    Arguments
    train_x     --  The training inputs.
    train_y     --  The training labels.
    mean_module --  The mean module. Defaults to a constant mean.
    covar_module--  The covariance module. Defaults to a RBF kernel.
    likelihood  --  The likelihood function. Defaults to Gaussian.
    """

    def __init__(
            self,
            train_x,
            train_y,
            mean_module=gpytorch.means.ConstantMean(batch_shape=torch.Size([556])),
            covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel()),
            likelihood=gpytorch.likelihoods.GaussianLikelihood(
                noise_constraint=gpytorch.constraints.GreaterThan(0.0)
            )
    ):
        super().__init__(train_x, train_y, likelihood)
        self.mean_module = mean_module
        self.covar_module = covar_module

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)

class ExactGPModel(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood):
        super(ExactGPModel, self).__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ConstantMean(batch_shape=torch.Size([556]))
        self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel())#batch_shape=torch.Size([556])))

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)

        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)
def train():
    model = ExactGPModel(train_x, train_y, likelihood)

    # train(model, train_x, train_y, n_iter=5)
    # Find optimal model hyperparameters
    model.train()
    likelihood.train()

    # Use the adam optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=0.1)  # Includes GaussianLikelihood parameters

    # optimizer = torch.optim.RAdam(model.parameters())  # Includes GaussianLikelihood parameters

    # "Loss" for GPs - the marginal log likelihood
    mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)

    var = 4

    training_iter = 10
    for i in range(training_iter):#, desc='Training iters'):
        # Zero gradients from previous iteration
        optimizer.zero_grad()
        # Output from model
        output = model(train_x)
        # print(output.shape)
        # Calc loss and backprop gradients
        loss = -mll(output, train_y[:, var]).sum()
        loss.backward()
        print('Iter %d/%d - Loss: %.3f' % (i + 1, training_iter, loss.item()))
        optimizer.step()

    cloudpickle.dump(model, open("gp_model.pickle", 'wb'))
    print("Saved")
def valid():
    model = cloudpickle.load(open("gp_model.pickle", 'rb'))
    batch = trs[-1]
    sqldloader = DataLoader(batch, num_workers=0,
        batch_sampler=tdata.BatchSampler(tdata.SequentialSampler(batch), batch_size=5, drop_last=False), #
        collate_fn=identity)
    valid_x, valid_y = next(iter(sqldloader))
    # Make predictions
    model.eval()
    with torch.no_grad():
        observed_pred = likelihood(model(valid_x))
        # Get mean
        mean = observed_pred.mean
        print(mean)
        # Get lower and upper confidence bounds
        lower, upper = observed_pred.confidence_region()

    f, ((y1_ax, y2_ax), (y3_ax, y4_ax)) = plt.subplots(2, 2, figsize=(8, 8))
    # Plot training data as black stars
    y1_ax.plot(train_x[0].detach().numpy(), train_y[0].detach().numpy(), 'k*')
    # Predictive mean as blue line
    y1_ax.plot(valid_x[0].squeeze().numpy(), mean[0, :].numpy(), 'b')
    # Shade in confidence
    y1_ax.fill_between(valid_x[0].squeeze().numpy(), lower[0, :].numpy(), upper[0, :].numpy(), alpha=0.5)
    y1_ax.set_ylim([-3, 3])
    y1_ax.legend(['Observed Data', 'Mean', 'Confidence'])
    y1_ax.set_title('Observed Values (Likelihood)')

    y2_ax.plot(train_x[1].detach().numpy(), train_y[1].detach().numpy(), 'k*')
    y2_ax.plot(valid_x[1].squeeze().numpy(), mean[1, :].numpy(), 'b')
    y2_ax.fill_between(valid_x[1].squeeze().numpy(), lower[1, :].numpy(), upper[1, :].numpy(), alpha=0.5)
    y2_ax.set_ylim([-3, 3])
    y2_ax.legend(['Observed Data', 'Mean', 'Confidence'])
    y2_ax.set_title('Observed Values (Likelihood)')

    y3_ax.plot(train_x[2].detach().numpy(), train_y[2].detach().numpy(), 'k*')
    y3_ax.plot(valid_x[2].squeeze().numpy(), mean[2, :].numpy(), 'b')
    y3_ax.fill_between(valid_x[2].squeeze().numpy(), lower[2, :].numpy(), upper[2, :].numpy(), alpha=0.5)
    y3_ax.set_ylim([-3, 3])
    y3_ax.legend(['Observed Data', 'Mean', 'Confidence'])
    y3_ax.set_title('Observed Values (Likelihood)')

    y4_ax.plot(train_x[3].detach().numpy(), train_y[3].detach().numpy(), 'k*')
    y4_ax.plot(valid_x[3].squeeze().numpy(), mean[3, :].numpy(), 'b')
    y4_ax.fill_between(valid_x[3].squeeze().numpy(), lower[3, :].numpy(), upper[3, :].numpy(), alpha=0.5)
    y4_ax.set_ylim([-3, 3])
    y4_ax.legend(['Observed Data', 'Mean', 'Confidence'])
    y4_ax.set_title('Observed Values (Likelihood)')
    plt.show()

valid()