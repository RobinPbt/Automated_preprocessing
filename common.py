# ---------------------------------------------------------------------------------------------------------------
# All imports necessary for automated_processing

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import itertools
import timeit

from PIL import Image
from abc import ABC

# Keras and spacy (for embeddings)
import spacy
import keras
from keras.preprocessing.text import one_hot,Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Dense, Flatten, Embedding, Input, Conv1D, GlobalMaxPooling1D, GlobalAveragePooling1D
from keras.models import Model

# Preprocessing
from sklearn.model_selection import train_test_split
from sklearn import manifold, decomposition
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder
from sklearn.preprocessing import StandardScaler, KBinsDiscretizer, Normalizer, PolynomialFeatures, PowerTransformer, RobustScaler
import imblearn

# Sklearn models
from sklearn.ensemble import RandomForestClassifier
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC

# Evaluation
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.metrics import log_loss
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import reciprocal

# ---------------------------------------------------------------------------------------------------------------
# Common functions

def get_types(x):
    """
    Function which categorize variables of a pandas.DataFrame between :
    - continuous_numerical_var (continuous numerical variables)
    - discrete_numerical_var (discrete numerical variables)
    - low_card_categorical_var (low cardinality categorical variables)
    - high_card_categorical_var (high cardinality categorical variables)
    
    Parameters
    ----------
    - x : dataframe to categorize (pandas.DataFrame)
    """
    
    categorical_var = [var for var in x.columns if x[var].dtype == 'object']
    numerical_var = list(x.drop(categorical_var, axis=1).columns)
    
    continuous_numerical_var = []
    discrete_numerical_var = []

    for var in numerical_var:
        ratio_unique = x[var].nunique() / len(x[var])
        if ratio_unique > 0.25: # ARBITRARY THRESHOLD FOR NOW, POSSIBLE TO IMPRVE DEFINITION ?
            continuous_numerical_var.append(var)
        else:
            discrete_numerical_var.append(var)
            
    low_card_categorical_var = []
    high_card_categorical_var = []

    for var in categorical_var:
        ratio_unique = x[var].nunique() / len(x[var])
        if ratio_unique > 0.25: # ARBITRARY THRESHOLD FOR NOW, POSSIBLE TO IMPRVE DEFINITION ?
            high_card_categorical_var.append(var)
        else:
            low_card_categorical_var.append(var)
    
    return continuous_numerical_var, discrete_numerical_var, low_card_categorical_var, high_card_categorical_var

def classifiers_test(x_train, y_train, x_valid, y_valid, models_dict):
    """
    Function which test a bunch of sklearn classification models 
    without hyperparameters optimization and return results on some standard metrics on a validation set.
    
    
    Parameters
    ----------
    - x_train : matrix of training inputs (array-like)
    - y_train : vector of training labels (array-like)
    - x_valid : matrix of training inputs (array-like)
    - y_valid : vector of training labels (array-like)
    - models_dict : dict of models to test with name as key and model as value, 
    ex : {"Random Forest" : RandomForestClassifier(random_state=5)}
    - random_state : RandomState instance (int)
    - max_iter : Maximum number of iterations taken for the solvers to converge (int)
    - n_jobs  : Number of CPU cores used when parallelizing over classes (int)
    """
    
    # Creating a df to store results on tested models
    results_df = pd.DataFrame()
    
    for model in models_dict.keys():
        clf = models_dict[model]
            
        # Train model and compute time
        start_time = timeit.default_timer()
        clf.fit(x_train, y_train)
        fit_time = timeit.default_timer() - start_time

        # Make predictions and compute time
        start_time = timeit.default_timer()
        predictions = clf.predict(x_valid)
        predict_time = timeit.default_timer() - start_time

        probas = clf.predict_proba(x_valid)[:,1]

        # Compute scores
        start_time = timeit.default_timer()
        accuracy = accuracy_score(y_valid, predictions)
        f1 = f1_score(y_valid, predictions)
        precision = precision_score(y_valid, predictions)
        recall = recall_score(y_valid, predictions)
        roc_auc = roc_auc_score(y_valid, probas)
        cross_entropy = log_loss(y_valid, predictions)
        compute_score_time = timeit.default_timer() - start_time

        # Store in df
        results_df[model] = [accuracy, f1, precision, recall, roc_auc, cross_entropy, fit_time, predict_time, compute_score_time]
    
    results_df.index = ['accuracy', 'f1', 'precision', 'recall', 'roc_auc', 'cross_entropy', 'fit_time', 'predict_time', 'compute_score_time']
    return results_df

def init_test(x_train, y_train, x_valid, y_valid, score, models_dict):
    """Initialize the dataframe to compare test results"""
    
    # DataFrame to store our results
    results_df = pd.DataFrame(index=[
        "Mean {}".format(score), 
        "Diff vs. initial", 
        "Max {}".format(score), 
        "Diff vs. initial",
        "Average diff", 
        "Keep transform", 
        "Initial variable",
    ])
    
    # Compute scores before transformation
    initial_scores = classifiers_test(x_train, y_train, x_valid, y_valid, models_dict=models_dict).loc[score]
    results_df["initial_scores"] = [np.mean(initial_scores), np.nan, np.max(initial_scores), np.nan, np.nan, np.nan, np.nan]
    
    return results_df, initial_scores

def test_keep_transform(initial_scores, new_scores):
    """Define wether a transformation should be kept regarding comparizon with initial results"""
    
    # Test wether transform had a positive impact
    if np.mean(new_scores) > np.mean(initial_scores) and np.max(new_scores) >= np.max(initial_scores):
        keep_transform = True
    else :
        keep_transform = False
        
    return keep_transform

def find_best_result(results_df, var):
    """After testing transformation/imputation/encoding define the best result regarding performance for each variable"""
    
    # Locate columns with the considered variable
    var_index = results_df.loc["Initial variable"][results_df.loc["Initial variable"] == var].index
    var_df = results_df[var_index]
    
    # Locate best result
    best_result_idx = var_df.loc["Average diff"].sort_values(ascending=False).index[0]
    
    # If best result is better than initial variable we keep it, else we keep initial variable
    if var_df[best_result_idx].loc["Keep transform"]:
        best_result = best_result_idx
    else:
        best_result = "initial variable"
        
    return best_result

def find_positives_results(results_df):
    """After testing combinations define results with positive impact on performance"""
    
    # Locate columns with True on Keep transform
    var_index = results_df.loc["Keep transform"][results_df.loc["Keep transform"] == True].index
    var_df = results_df[var_index]
    
    return var_df

def nan_inf_replace(x, var):
    x[var].replace([np.inf, -np.inf], np.nan, inplace=True)
    x[var].fillna(0, inplace=True)
    
    return x

# ---------------------------------------------------------------------------------------------------------------
# Parent class to all testers

class Tester(ABC):
    """Parent class of testers"""
    
    def __init__(self, variables, score, models_dict, train_size=0.8, random_state=None):
        self.variables = variables
        self.score = score
        self.models_dict = models_dict
        self.random_state = random_state
        self.results_df = None
        self.best_results_df = None
        self.positive_results_df = None
        self.train_size = train_size
        self.is_fitted = False
        self.x_fitted = None
        self.y_fitted = None
        
    def fit(self, x, y):
        
        # IMPORTANT NOTE : the fitting step based on a train / valid process but for transform we will keep the entire fitted matrix as reference
        self.is_fitted = True
        self.x_fitted = x.copy()
        self.y_fitted = y.copy()
    
    def transform(self, x):
        
        # Test if object has been fitted first
        if not self.is_fitted:
            raise NotImplementedError("{} hasn't been fitted. You must fit before transforming".format(self))
    
    def fit_transform(self, x, y):
        """Apply fit and then transform function"""
        
        self.fit(x, y)
        x = self.transform(x)
        
        return x