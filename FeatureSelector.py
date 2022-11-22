from common import *

# Unsupervised methods
from sklearn.feature_selection import VarianceThreshold
from statsmodels.stats.outliers_influence import variance_inflation_factor

# Wrappers methods
from sklearn.feature_selection import SequentialFeatureSelector
from sklearn.feature_selection import RFE

# Filter methods
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_regression
from sklearn.feature_selection import r_regression
from sklearn.feature_selection import chi2
from sklearn.feature_selection import mutual_info_regression
from sklearn.feature_selection import mutual_info_classif
from sklearn.feature_selection import f_classif
from scipy.stats import spearmanr
from scipy.stats import kendalltau
from scipy.stats import pointbiserialr

# ---------------------------------------------------------------------------------------------------------------
# FeatureSelector class

class FeatureSelector(Tester):
    """
    Class which apply different feature selection methods : unsupervised, wrapper and filter on numerical variables of x matrix to infer the best set of features. 
    Final feature selection is based on a vote : features retained in a percentage of methods superior to a given threshold will be selected. 
    
    Parameters
    ----------
    - models_dict : dict of models to test with name as key and model as value, 
    ex : {"Random Forest" : RandomForestRegressor(random_state=5)}. Note that only wrappers methods use these models
    - ML_type : type of machine learning algorithm, can be 'Regression' or 'Classification' (str)
    - random_state : RandomState instance (int)
    - threshold : Minimal percentage of votes a feature must get to be selected (float in range [0:1])  
    - methods_list : features selection methods to test (list)
    
    Possible methods_list values are :
    - Unsupervised methods : 'Variance Threshold', 'VIF Threshold'
    - Wrapper methods : 'Backward Selection', 'Forward Selection', 'Recursive Feature Elimination' 
    (for RFE method, models in models_dict should have 'coef_' or 'feature_importances_' attribute)
    - Filter methods, regression : 'Filter Pearson Coefficient', 'Filter Spearman Rho', 'Filter Kendall Tau', 'Filter Mutual information reg'
    - Filter methods, classification : 'Filter Point-biserial correlation', 'Filter Chi2', 'Filter ANOVA F-score', 'Filter Mutual information clas'
    ex: ['Variance Threshold', 'Recursive Feature Elimination', 'Filter Pearson Coefficient', 'Filter Kendall Tau'] for a regression problem
    """
    
    def __init__(self, models_dict, ML_type, methods_list, threshold=0.25):
        self.models_dict = models_dict
        self.ML_type = ML_type
        self.methods_list = methods_list
        self.threshold = threshold
        self.votes = None
        self.selected_variables = None
        self.is_fitted = False
        self.x_fitted = None
        self.y_fitted = None
        
        # Test if methods_list is correct      
        possible_regression_list = [
            'Variance Threshold', 'VIF Threshold', 
            'Backward Selection', 'Forward Selection', 'Recursive Feature Elimination',
            'Filter Pearson Coefficient', 'Filter Spearman Rho', 'Filter Kendall Tau', 'Filter Mutual information reg',
        ]
        
        possible_classification_list = [
            'Variance Threshold', 'VIF Threshold', 
            'Backward Selection', 'Forward Selection', 'Recursive Feature Elimination',
            'Filter Point-biserial correlation', 'Filter Chi2', 'Filter ANOVA F-score', 'Filter Mutual information clas'
        ]
    
        for method in self.methods_list:
            if self.ML_type == 'Regression':
                if method not in possible_regression_list:
                    raise ValueError("Method name '{}' is incorrect, possible values for regression are {}".format(method, possible_regression_list))
            elif self.ML_type == 'Classification':
                if method not in possible_classification_list:
                    raise ValueError("Method name '{}' is incorrect, possible values for classification are {}".format(method, possible_classification_list))
                

    def fit(self, x, y):
        
        # Initialize df
        votes = pd.DataFrame(columns=x.columns)
        
        # Follow the Google rule of thumb of approx 100 training examples for 1 feature
        n_features_to_select = round(x.shape[0] / 100)
        
        # Correspondance between methods_list and functions
        unsupervised_dict = {
            'Variance Threshold' : variance_threshold,
            'VIF Threshold' : vif_threshold,
        } 
        
        wrappers_dict = {
            'Backward Selection' : backward_selection,
            'Forward Selection' : forward_selection,
            'Recursive Feature Elimination' : recursive_feature_elimination,
        } 

        filters_dict = {
            'Filter Pearson Coefficient' : pearson_filter,
            'Filter Spearman Rho' : spearman_filter,
            'Filter Kendall Tau' : kendall_filter,
            'Filter Mutual information reg' : mutual_info_reg_filter,
            'Filter Point-biserial correlation' : point_biserial_filter,
            'Filter Chi2' : chi2_filter,
            'Filter ANOVA F-score' : anova_filter,
            'Filter Mutual information clas' : mutual_info_clas_filter,
        } 
        
        methods_count = 0
        
        # Perform features selection on unsupervised methods
        for method in self.methods_list:
            if method in unsupervised_dict.keys():
                features_kept = unsupervised_dict[method](x)
                self.votes = add_selector_results(self.votes, x, features_kept, method)
                methods_count += 1
                
        # Perform features selection on wrappers methods
        for method in self.methods_list:
            if method in wrappers_dict.keys():
                for model_name in self.models_dict.keys(): # For wrappers method we need to test them for each model
                    model = models_dict[model_name]
                    features_kept = wrappers_dict[method](x, y, model, n_features_to_select)
                    self.votes = add_selector_results(self.votes, x, features_kept, "{}_{}".format(method, model_name))
                    methods_count += 1
        
        # Perform features selection on filter methods
        for method in self.methods_list:
            if method in filters_dict.keys():
                features_kept = filters_dict[method](x, y, n_features_to_select)
                self.votes = add_selector_results(self.votes, x, features_kept, method)
                methods_count += 1
        
        # We keep each feature with a selection percentage above or equal to the threshold
        self.selected_variables = (self.votes.sum(axis=0) / methods_count)[(self.votes.sum(axis=0) / methods_count) >= self.threshold].index
        
        super().fit(x, y)
        
    def transform(self, x):
        super().transform(x)
        return x[self.selected_variables]
                
                
# ---------------------------------------------------------------------------------------------------------------
# Functions related to the class           

def add_selector_results(votes, x, features_kept, method):
    """Add results to vote dataframe"""
    
    results = [1 if var in features_kept else 0 for var in x.columns]
    results = pd.DataFrame(np.array(results).reshape(1,-1), columns=x.columns, index=[method])
    votes = pd.concat([votes, results], axis=0)
    return votes

# Unsupervised methods

def variance_threshold(x, threshold=0.01):
    """Delete variables with variance under a given threshold"""
    
    selector = VarianceThreshold(threshold=threshold)
    selector.fit(x)
    return selector.get_feature_names_out()

def vif_threshold(x, threshold=5):
    """Delete variables with VIF over a given threshold"""
    
    vif_scores = pd.DataFrame()
    vif_scores["feature"] = x.columns
    vif_scores["VIF"] = [variance_inflation_factor(x.values, i) for i in range(len(x.columns))]
    vif_scores = vif_scores.replace(np.inf, 99) # If really strong independance (R2 = 1) VIF creates inf values  
    return list(vif_scores[vif_scores["VIF"] < 5]["feature"].values)
    
# Wrappers methods

def backward_selection(x, y, model, n_features_to_select):
    """Select the best subset of variables on a given model with backward selection"""
    
    # TO DO : ADAPT WITH PERSONNALIZED SCORER
    selector = SequentialFeatureSelector(model, n_features_to_select=n_features_to_select, direction='backward')
    selector.fit(x, y)
    return selector.get_feature_names_out()

def forward_selection(x, y, model, n_features_to_select):
    """Select the best subset of variables on a given model with forward selection"""
    
    # TO DO : ADAPT WITH PERSONNALIZED SCORER
    selector = SequentialFeatureSelector(model, n_features_to_select=n_features_to_select, direction='forward')
    selector.fit(x, y)
    return selector.get_feature_names_out()

def recursive_feature_elimination(x, y, model, n_features_to_select):
    """Select the best subset of variables on a given model with recursive feature elimination"""
    
    selector = RFE(model, n_features_to_select=n_features_to_select)
    selector.fit(x, y)
    return selector.get_feature_names_out()

# Filter methods : regression

def pearson_filter(x, y, n_features_to_select):
    """Return most correlated variables to target taking Pearson's R 
    (linear relationship between 2 numerical variables)"""

    selector = SelectKBest(r_regression, k=n_features_to_select)
    selector.fit(x, y)
    return selector.get_feature_names_out()

def spearman_filter(x, y, n_features_to_select):
    """Return most correlated variables to target taking Spearman’s Rho 
    (non linear relationship between 2 numerical variables, 0 indicates no correlation and -1/1 strong negative/positive correlation)"""
    
    coeffs = {var : abs(spearmanr(x[var], y).correlation) for var in x.columns} # Compute absolute correlations values
    coeffs_df = pd.Series(coeffs).sort_values(ascending=False) # Sorting values
    return coeffs_df[:n_features_to_select].index # Keeping k bests

def kendall_filter(x, y, n_features_to_select):
    """Return most correlated variables to target taking Kendall’s Tau 
    (non linear relationship between 2 numerical variables, -1 indicates no correlation and 1 strong correlation)"""
    
    coeffs = {var : kendalltau(x[var], y).correlation for var in x.columns} # Compute correlations values
    coeffs_df = pd.Series(coeffs).sort_values(ascending=False) # Sorting values
    return coeffs_df[:n_features_to_select].index # Keeping k bests

def mutual_info_reg_filter(x, y, n_features_to_select):
    """Return most dependant variables to target taking mutual information 
    (dependancy between two random variables, 0 indicates no dependence and 1 strong dependence)"""

    selector = SelectKBest(mutual_info_regression, k=n_features_to_select)
    selector.fit(x, y)
    return selector.get_feature_names_out()

# Filter methods : classification

def point_biserial_filter(x, y, n_features_to_select):
    """Return most correlated variables to target taking point-biserial 
    (relationship between a binary and a continuous variable, 0 indicates no correlation and -1/1 strong negative/positive correlation)"""
    
    coeffs = {var : abs(pointbiserialr(x[var], y).correlation) for var in x.columns} # Compute absolute correlations values
    coeffs_df = pd.Series(coeffs).sort_values(ascending=False) # Sorting values
    return coeffs_df[:n_features_to_select].index # Keeping k bests

def chi2_filter(x, y, n_features_to_select):
    """Return most dependant variables to target taking Chi2 
    (dependency between two nominal variables)"""

    selector = SelectKBest(chi2, k=n_features_to_select)
    selector.fit(x, y)
    return selector.get_feature_names_out()

def anova_filter(x, y, n_features_to_select):
    """Return most related variables to target taking ANOVA F-score 
    (relationship between a continuous and a nominal variable)"""

    selector = SelectKBest(f_classif, k=n_features_to_select)
    selector.fit(x, y)
    return selector.get_feature_names_out()

def mutual_info_clas_filter(x, y, n_features_to_select):
    """Return most dependant variables to target taking mutual information 
    (dependancy between two random variables, 0 indicates no dependence and 1 strong dependence)"""

    selector = SelectKBest(mi_class_corr, k=n_features_to_select)
    selector.fit(x, y)
    return selector.get_feature_names_out()