## [Apache Spark](https://spark.apache.org)

**Apache Spark™** is a multi-language engine for executing data engineering, data science, and machine learning on single-node machines or clusters.

## [Shap]()
https://www.youtube.com/watch?v=ZkIxZ5xlMuI
Explainable models/hyperparams

## [Vowpal Wabbit](https://vowpalwabbit.org/index.html)
Vowpal Wabbit provides fast, efficient, and flexible online machine learning techniques for reinforcement learning, supervised learning, and more.

## Hyperopt
Hyperparam tuning
## [Optuna](https://optuna-dashboard.readthedocs.io/en/latest/index.html)
Hyperparam tuning, supposedly better than Hyperopt because of better search strategy (bayesian optimisastion)
# [CuPY](https://cupy.dev)
Numpy but with CUDA

## [Polars](https://pola.rs)
Better Pandas (Lazy loading vs eager)
https://www.kaggle.com/competitions/leap-atmospheric-physics-ai-climsim/discussion/495128

# Data Formats
[From stackoverflow:](https://stackoverflow.com/questions/37928794/which-is-faster-for-load-pickle-or-hdf5-in-python)
- **Parquet**
    - **pros**
        - one of the fastest and widely supported **binary** storage formats
        - supports very fast compression methods (for example Snappy codec)
        - de-facto standard storage format for Data Lakes / BigData
    - **contras**
        - the whole dataset must be read into memory. You can't read a smaller subset. One way to overcome this problem is to use **partitioning** and to read only required partitions.
            - no support for indexing. you can't read a specific row or a range of rows - you always have to read the whole Parquet file
        - Parquet files are **immutable** - you can't change them (no way to append, update, delete), one can only either write or overwrite to Parquet file. Well this "limitation" comes from the BigData and would be considered as one of the huge "pros" there.
- **HDF5** ([kaggle](https://www.kaggle.com/competitions/leap-atmospheric-physics-ai-climsim/discussion/497304))
    - **pros**
        - supports data slicing - ability to read a portion of the whole dataset (we can work with datasets that wouldn't fit completely into RAM).
        - relatively fast **binary** storage format
        - supports compression (though the compression is slower compared to Snappy codec (Parquet) )
        - supports appending rows (mutable)
    - **contras**
        - [risk of data corruption](https://cyrille.rossant.net/moving-away-hdf5/)
- **Pickle**
    - **pros**
        - very fast
    - **contras**
        - requires much space on disk
        - for a long term storage one might experience compatibility problems. You might need to specify the Pickle version for reading old Pickle files.
- Feather (Apache Arrow)