# KNN-Hyperparameter-Tuner

K-Nearest Neighbors (k-NN or KNN) is a machine learning algorithm that is supervised and non-parametric used for classification and regression for statistical inference. It requires a dataset of labelled datapoints of any dimensionality and hyperparameters. It is a lazy learning algorithm which does not build a model or representation to decide inferences but rather requires appropriate hyperparameters to define algorithm behavior of k-value, distance metric, and weighting scheme. This project searches for performant k-NN hyperparameter values. This requires a testing dataset and a k-value for k-fold cross validation (CV) or an additional test dataset. Once the hyperparameters are known then k-NN can run inference on any data any number of times immediately.

## k-NN Hyperparameters

+ k-NN k-value: an integer between 1 and the number of total datapoints
+ Distance Metric: a function of two data points of arbitrary dimension
+ Weight Scheme: a weight function of the distance

## Hyperparameter Search Parameters

+ Training dataset: a set of labelled data points of arbitrary dimension
+ Test dataset OR
+ CV k-value: an integer

## TODO

+ Implement k-fold cross validation
+ Search for weighting method
+ Replace brute force search with random search

## License

This project is available for personal and commercial use under the [MIT license](LICENSE).
