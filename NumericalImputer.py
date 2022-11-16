from common import *

# ---------------------------------------------------------------------------------------------------------------
# NumericalImputer class

class NumericalImputer(Tester):
    """
    Class which apply different imputations on numerical variables of x matrix and test these imputations on some models to infer the best imputations
    given models scores. The scoring is performed on a validation set created from the fitted matrice."" 
    
    Parameters
    ----------
    - variables : name of variables to be processed, must be numerical variables (list)
    - score : score to be optimized (XX)
    - models_dict : dict of models to test with name as key and model as value, 
    ex : {"Random Forest" : RandomForestRegressor(random_state=5)}
    - ML_type : type of machine learning algorithm, can be 'Regression' or 'Classification' (str)
    - train_size : percentage of the dataset used as a training set (float in range [0:1])
    - random_state : RandomState instance (int)
    - imputation_list : imputation to test (list)
    
    Possible values are ['mean', 'median', 'most_frequent', 'constant']
    """
    
    def __init__(self, variables, score, models_dict, ML_type, train_size=0.8, random_state=None, imputation_list=['mean', 'median', 'most_frequent', 'constant']):
        self.imputation_list = imputation_list
        super().__init__(variables, score, models_dict, ML_type, train_size, random_state)
        
        # Test if transformation_list is correct
        possible_list = ['mean', 'median', 'most_frequent', 'constant']
    
        for imputation in self.imputation_list:
            if imputation not in possible_list:
                raise ValueError("Imputation name '{}' is incorrect, possible values are {}".format(imputation, possible_list))
        
    def fit(self, x, y):
        """
        Fitting NumericalImputer with x and y matrices to find best imputations. 
        Note that initial scores (base reference) corresponds to scores with missing values filled by 0.
        """ 
        
        # Split between train and validation set 
        x_train, x_valid, y_train, y_valid = train_test_split(x, y, train_size=self.train_size, random_state=self.random_state)
        
        # Define if the score must be maximized or minimized
        score_strategy = find_score_optmization_strategy(self.score)
        
        # DataFrame to store our results, we fill NaN values by 0 for initial scores
        self.results_df, initial_scores = init_test(
            x_train[self.variables].fillna(0), y_train,
            x_valid[self.variables].fillna(0), y_valid, 
            self.score, self.models_dict, self.ML_type, score_strategy
        )
        
        # We test each imputation on each variable
        for var in self.variables:
            if x_train[var].isna().sum() == 0 and x_valid[var].isna().sum() == 0: 
                pass # If no missing values we don't need to assess imputation
            else:
                for imputation in self.imputation_list:
                    self.results_df = var_impute_test(
                        x_train[self.variables], y_train, 
                        x_valid[self.variables], y_valid, 
                        var, initial_scores, score_strategy, self.results_df, self.models_dict, self.ML_type, self.score, impute=imputation
                    )
                
        # We save best imputation for each variable
        self.best_results_df = pd.DataFrame(index=["Best result"])

        for var in self.variables:
            if x_train[var].isna().sum() == 0 and x_valid[var].isna().sum() == 0: 
                pass
            else:
                self.best_results_df[var] = find_best_result(self.results_df, var, score_strategy)
            
        super().fit(x, y)
        
    def transform(self, x):
        """
        Transform a x matrix with best imputations found on fitting step. 
        These imputations are computed after the initial matrix used during fitting.
        """
        
        super().transform(x)

        # Apply best imputations. Imputation is first fitted with x_fitted matrix and then apply to the matrix specified in transform
        for var in self.best_results_df.columns:
            val = self.best_results_df[var].values[0]
            val = val.split("_")

            if val[0] == "mean":
                imputer = SimpleImputer(strategy="mean")
                imputer.fit(self.x_fitted[var].values.reshape(-1, 1))
                x[var] = imputer.transform(x[var].values.reshape(-1, 1))
            elif val[0] == "median":
                imputer = SimpleImputer(strategy="median")
                imputer.fit(self.x_fitted[var].values.reshape(-1, 1))
                x[var] = imputer.transform(x[var].values.reshape(-1, 1))
            elif val[0] == "most":
                imputer = SimpleImputer(strategy="most_frequent")
                imputer.fit(self.x_fitted[var].values.reshape(-1, 1))
                x[var] = imputer.transform(x[var].values.reshape(-1, 1))
            elif val[0] == "constant":
                imputer = SimpleImputer(strategy="constant")
                imputer.fit(self.x_fitted[var].values.reshape(-1, 1))
                x[var] = imputer.transform(x[var].values.reshape(-1, 1))
            elif val[0] == "initial variable":
                x[var].fillna(0, inplace=True)

        return x

# ---------------------------------------------------------------------------------------------------------------
# Functions related to the class

def apply_impute(x_train, strategy, var, x_valid=pd.DataFrame()):
    
    imputer = SimpleImputer(strategy=strategy)
    x_train[var] = imputer.fit_transform(x_train[var].values.reshape(-1, 1))
    x_valid[var] = imputer.transform(x_valid[var].values.reshape(-1, 1))
    
    return x_train, x_valid

def var_impute_test(x_train, y_train, x_valid, y_valid, var, initial_scores, score_strategy, results_df, models_dict, ML_type, score, impute):
    
    x_train_bis = x_train.copy()
    x_valid_bis = x_valid.copy()

    # Proceed imputation on given variable
    x_train_bis, x_valid_bis = apply_impute(x_train=x_train_bis, strategy=impute, var=var, x_valid=x_valid_bis)

    # For other variables, fill missing values by 0
    x_train_bis.fillna(0, inplace=True)
    x_valid_bis.fillna(0, inplace=True)

    # Computing and comparing results of imputation
    if ML_type == "Classification":
        new_scores = classifiers_test(x_train_bis, y_train, x_valid_bis, y_valid, models_dict=models_dict).loc[score]
    elif ML_type == "Regression":
        new_scores = regressors_test(x_train_bis, y_train, x_valid_bis, y_valid, models_dict=models_dict).loc[score]

    keep_transform = test_keep_transform(initial_scores, new_scores, score_strategy)

    # Saving results
    KPIs_scores = get_KPIs(initial_scores, new_scores, score_strategy, keep_transform, var)
    results_df["{}_imputed_{}".format(impute, var)] = KPIs_scores
    
    return results_df