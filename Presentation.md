## Models

### 1. ClimaX (Microsoft)

* Uses a trained ***transformer*** along with input variable encoding to handle any  combination of data and task
* Trained on multiple datasets
* Pre-trained network available for download
* Fine-Tuning for multiple outputs at once without significant loss of precision

## 2. XGBoost

* Builds a decision tree forest for each target variable


## Reviewed but not directly applicable:

* [[2022Correcting Coarse‐Grid Weather and Climate Models by Machine Learning From.pdf]]
	* ## Use ML to predict model error and correct it
	*  ***Nudging*** the outputs closer to their actual values (from ground truth observation) during runs
	* Use ML to learn the nudge tendencies and apply them to the coarse-grained model (to correct it)
	* Train different Neural Nets for each target feature set (so hyperparams can be optimized accordingly)
	* Train multiple NNs for the same target and average the means
	* Requires time data
* [Could Machine Learning Break the Convection Parameterization Deadlock?](https://agupubs.onlinelibrary.wiley.com/doi/epdf/10.1029/2018GL078202)
	* An early implementation of ClimSim NN
	* Mentions NNs being too smooth, lacking randomness:
		>However, one issue for the convective heating and particularly moistening rates is that the NN predictions are smoother and do not exhibit as much of the variability as SP-CAM (internal stochastic variability). Indeed, the ANN is by definition deterministic and thus cannot reproduce any stochasticity.

 * [Stable climate simulations using a realistic general circulation model with neural network parameterizations](file:///G:/Licenta/PDF/gmd-15-3923-2022.pdf)
	 * ResDNN > DNN
	 * Multiple NNs for different outputs to mitigate interference