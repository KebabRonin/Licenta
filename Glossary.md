## Meteorology

**Convection**: 
> **Convection** is single or [multiphase](https://en.wikipedia.org/wiki/Multiphase_flow "Multiphase flow") [fluid flow](https://en.wikipedia.org/wiki/Fluid_flow "Fluid flow") that occurs [spontaneously](https://en.wikipedia.org/wiki/Spontaneous_process "Spontaneous process") due to the combined effects of [material property](https://en.wikipedia.org/wiki/Material_property "Material property") [heterogeneity](https://en.wikipedia.org/wiki/Heterogeneity "Heterogeneity") and [body forces](https://en.wikipedia.org/wiki/Body_forces "Body forces") on a [fluid](https://en.wikipedia.org/wiki/Fluid)

**Evapotranspiration**:
>  **Evapotranspiration** (**ET**) refers to the combined processes which move water from the Earth's surface (open water and ice surfaces, bare soil and [vegetation](https://en.wikipedia.org/wiki/Vegetation "Vegetation")) into the [atmosphere](https://en.wikipedia.org/wiki/Atmosphere_of_Earth "Atmosphere of Earth").  It covers both water [evaporation](https://en.wikipedia.org/wiki/Evaporation "Evaporation") (movement of water to the air directly from soil, [canopies](https://en.wikipedia.org/wiki/Canopy_interception "Canopy interception"), and water bodies) and [transpiration](https://en.wikipedia.org/wiki/Transpiration "Transpiration") (evaporation that occurs through the stomata, or openings, in plant leaves). Evapotranspiration is an important part of the local [water cycle](https://en.wikipedia.org/wiki/Water_cycle "Water cycle") and [climate](https://en.wikipedia.org/wiki/Climate "Climate"), and measurement of it plays a key role in [agricultural irrigation](https://en.wikipedia.org/wiki/Irrigation "Irrigation") and [water resource management](https://en.wikipedia.org/wiki/Water_resource_management "Water resource management").
### Meteo Models

**Nudging**: [Correcting Weather and Climate Models by Machine Learning Nudged Historical Simulations - Watt‐Meyer - 2021 - Geophysical Research Letters - Wiley Online Library](https://agupubs.onlinelibrary.wiley.com/doi/10.1029/2021GL092555)
influence prediction towards actual observed values to estimate state-dependant biases (**requires time series**)

**Global Climate Model**

**Cloud Resolving Model**

**Large Eddy Simulation**

**Parametrization**: 
> 1. Representation of a (distribution) function using a number of weights (parameters)
> 2. A parameterization is a method for estimating parameter values without explicitly simulating the processes directly. Parameterizations can sometimes be thought of as very low order models.

**Super-Parametrization**:
> 1. Super-parameterization replaces one or more parameterizations with a second model that is designed to simulate the processes explicitly, in order to provide more accurate parameter values back to the main “host” model.

###### Example: ([Source](https://hannahlab.org/blog/what-is-super-parameterization/))

As a simple illustrative example, consider the typical Lotka-Volterra example of a ecosystem consisting of a population of rabbits (R) and foxes (F).

![](http://hannahlab.org/wp-content/uploads/2017/03/296.png)

A model of the system can be written,  
![](http://hannahlab.org/wp-content/uploads/2017/03/344.png)  
We have 4 parameters (_a,b,c,d_), and typically we would just assume constant values to explore the population dynamics. Alternatively, we could parameterize these values to represent other processes that are not in the basic model. For instance, maybe foxes are more efficient or successful hunters during summer. This would be simple to parameterize by making the parameter _b_ fluctuate with the day of the year.

Now what if we consider that foxes also become less efficient hunters as they age? This is difficult to parameterize if we don’t assume something about the age distribution of foxes. The MMF approach to this problem would be to use a secondary model that explicitly simulates the lifecycle and hunting efficiency of a small representative sample of foxes in the ecosystem. This would allow the age distribution of foxes to evolve with time, which could then be used to obtain an evolving estimate of the parameter _b_. Additionally, the fox model could be designed to affect the fox birth rate _c_, since only mature foxes can reproduce. This would be a “super-parameterization”.

## Machine Learning

**Affine Transformation**: preserves relationships between transformed variables (parallel lines remain parallel)

**Grid Search**: Hyperparameter optimization (try all combinations and find out)

**Gaussian Process**: A non-parametric, probabilistic model for regression, classification, and uncertainty quantification. It depicts a group of random variables, each of which has a joint Gaussian distribution and can have a finite number. They deliver predictions as probability distributions.

**Gaussian Process Regression**:

**out-of-fold prediction**: is a prediction by the model during the k-fold cross-validation procedure.
### Ensambles

> Use uncorrelated models, so the ensamble has more predictive power

>**GOSS**: Gradient-based One-Side Sampling -  keep only samples with steep gradient (for gradient descent) to consider for splits(Trees)
>
>**Bagging (Bootstrap Aggregation)**: Parallel training of models on the data, then take aggregate all results (weighted/max voting)
>
>**Random Forest**: Bagging, but each model only has access to a subset of features
>
>**Boosting**: Sequential chain of models: Model 1 predicts from data and has a residual, which is used as input for another model to predict some other part of the output + residual, etc
>
> **Gradient Boosting**: weight samples by gradient (gradient descent) steepness 
>
> [Source](https://www.youtube.com/watch?v=o7cUF25hAbo)

### Useless but still interesting:

**State Vector Machine**: Something-something The kernel trick. (!!!) Storing the kernel matrix requires memory that scales quadratically with the number of data points. Training time for traditional SVM algorithms also scales superlinearly with the number of data points. So, **these algorithms aren't feasible for large data sets**.

[**The Kernel Trick**](https://www.youtube.com/watch?v=Q7vT0--5VII): Apply a non-linear transformation to the dataset (to move it in a higher dimensional space) and fit the model on more dimensions => non-linear separation plane. The transformation doesn't need to be computed (!!!), only a ordering function needs to be defined (aka. the Kernel). **[Basically Neural Networks use neurons to find the higher dimensional transformation](https://sizhky.github.io/posts/2017/11/neural-network-transformations.html)**