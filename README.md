# KNN-Hyperparameter-Tuner

K-Nearest Neighbors (k-NN or KNN) is a machine learning algorithm that is supervised and non-parametric used for classification and regression for statistical inference. It requires a dataset of labelled datapoints of any dimensionality and hyperparameters. It is a lazy learning algorithm which does not build a model or representation to decide inferences but rather requires appropriate hyperparameters to define algorithm behavior of k-value, distance metric, and weighting scheme. This project searches for performant k-NN hyperparameter values. This requires a testing ddataset and k-fold cross validation (CV) which is also a hyperpameter or an additional test dataset.

## Hyperparameters

+ k-NN k-value: an integer between 1 and the number of total datapoints
+ Distance Metric: a function of two data points of the same arbitrary dimensionality
+ Weighting Schene: a uniform or weighted scheme with a weight function
+ CV k-value: as an integer

## TODO

+ Implement k-fold cross validation
+ Search for weighting method
+ Replace brute force search with random search
+ Option for hyperrectangles instead of hyperspheres

## References

1. [ProgrammingR: KNN Hyperparameters — A Friendly Guide to Optimization](https://www.programmingr.com/knn-hyperparameters-a-friendly-guide-to-optimization/)

## License

This project is available for personal and commercial use under the [MIT license](LICENSE).
