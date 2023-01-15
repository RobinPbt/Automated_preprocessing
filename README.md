# Automated_preprocessing

This repository contains classes to perform automated preprocessing for supervised regression or classification. Please note it is a pending project.

For each class, the main idea is to try a preprocessing action and to test if it improves performances vs. a baseline on a given set of models with a given evaluation metric. Parameters notably includes models to test, type of task (regression or classification), and the preprocessing actions to test.

Each class works with a fit (taking the X and y matrices as argument) and a transform (taking X matrix as argument) methods. The fit method returns the results of each tests and gives the best results. The transform method applies the best results. 

Please find below the different classes:
  -	NumericalImputer:
    -	Try several imputations methods for numerical variables
    -	Baseline: fill missing values by 0
    - Possible tests: mean, median, most_frequent, and constant imputations
  -	CategoricalImputerEncoder:
    -	Try several encoding and imputations methods for categorical variables
    -	Baseline: delete categorical variables
    -	Possible tests:
        -	Imputation: most_frequent, NaN as a category
        -	Encoding: one-hot, ordinal
  -	HighCategoricalEncoder:
    -	Try several encoding methods for categorical variables with high cardinality
    -	Baseline: delete categorical variables
    -	Possible tests: ordinal, embeddings
  -	NumericalTransformer:
    -	Try several transformations for numerical variables
    -	Baseline: variables without transformations
    -	Possible tests: log, discretize, standardize, normalize, power, robust_scale, square, sqrt, cos, tan, sin
  -	NumericalCombiner:
    -	Try several combinations on each couple of numerical variables
    -	Please note that you can run this class several times if you want to get more complex combinations (ex: sum of 2 variables divided by a third one)
    -	Baseline: variables without combinations
    -	Possible tests: sum, subtraction, division, multiplication
  -	FeatureSelector:
    -	Class which applies different feature selection methods: unsupervised, wrapper and filter on numerical variables
    -	Final feature selection is based on a vote: features retained in a percentage of methods superior to a given threshold will be selected
    -	Baseline: initial set of variables
    -	Possible tests: 
        -	Unsupervised methods: Variance Threshold, VIF Threshold
        -	Wrapper methods: Backward Selection, Forward Selection, Recursive Feature Elimination 
        -	Filter methods, regression: Filter Pearson Coefficient, Filter Spearman Rho, Filter Kendall Tau, Filter Mutual information reg
        -	Filter methods, classification: Filter Point-biserial correlation, Filter Chi2, Filter ANOVA F-score, Filter Mutual information

The FullPreprocessor class performs all these preprocessing tests in once by default, in the order presented above. You can also specify which tests you want to try.
