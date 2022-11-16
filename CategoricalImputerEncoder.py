from common import *

# ---------------------------------------------------------------------------------------------------------------
# CategoricalImputerEncoder class

class CategoricalImputerEncoder(Tester):
    """
    Class which apply different imputations and encodings on a x matrix and test these imputations on some models to infer the best imputations
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
    
    Possible values are ['most_frequent', 'nan_category']
    
    - encoding_list : endodings to test (list)
    
    Possible values are ['one-hot', 'ordinal']
    """
    
    def __init__(self, variables, score, models_dict, ML_type, train_size=0.8, random_state=None, imputation_list=['most_frequent', 'nan_category'], encoding_list=['one-hot', 'ordinal']):
        self.imputation_list = imputation_list
        self.encoding_list = encoding_list
        super().__init__(variables, score, models_dict, ML_type, train_size, random_state)
        
        # Test if imputation_list and encoding_list are correct
        possible_impute_list = ['most_frequent', 'nan_category']
        possible_encode_list = ['one-hot', 'ordinal']

        for impuation in self.imputation_list:
            if impuation not in possible_impute_list:
                raise ValueError("Imputation name '{}' is incorrect, possible values are {}".format(impuation, possible_impute_list))

        for encoding in self.encoding_list:
            if encoding not in possible_encode_list:
                raise ValueError("Encoding name '{}' is incorrect, possible values are {}".format(encoding, possible_encode_list))
        
    def fit(self, x, y):
        """
        Fitting CategoricalImputerEncoder with x and y matrices to find best imputations and encodings. 
        Note that initial scores (base reference) corresponds to scores without categorical variables.
        """ 
        
        # Get variables types
        continuous_numerical_var, discrete_numerical_var, low_card_categorical_var, high_card_categorical_var = get_types(x)
        numerical_var = continuous_numerical_var + discrete_numerical_var
        
        # Split between train and validation set 
        x_train, x_valid, y_train, y_valid = train_test_split(x, y, train_size=self.train_size, random_state=self.random_state)
        
        # DataFrame to store our results, we use only numerical values for initial scores
        self.results_df, initial_scores = init_test(
            x_train[numerical_var], y_train,
            x_valid[numerical_var], y_valid, 
            self.score, self.models_dict, self.ML_type
        )
        
        # We create possible combinations of impuation and encoding
        combinations = []

        for imputation in self.imputation_list:
            for encoding in self.encoding_list:
                combinations.append([imputation, encoding])

        # We test each combination on each variable
        for var in self.variables: 
            for combination in combinations:
                self.results_df = var_impute_encode_test(
                    x_train, y_train, 
                    x_valid, y_valid, 
                    var, initial_scores, self.results_df, self.models_dict, self.ML_type, self.score, combination=combination,
                )

        # We save best transformation for each variable
        self.best_results_df = pd.DataFrame(index=["Best result"])

        for var in self.variables: 
            self.best_results_df[var] = find_best_result(self.results_df, var)
        
        super().fit(x, y)
        
    def transform(self, x):
        """
        Transform a x matrix with best imputations and encoding found on fitting step. 
        These imputations and encodings are computed after the initial matrix used during fitting.
        """
        
        super().transform(x)
    
        # Apply best imputations and encodings. First fitted with x_fitted matrix and then apply to the matrix specified in transform
        for var in self.best_results_df.columns:
            val = self.best_results_df[var].values[0]
            val = val.split("_")

            # First test if variable doesn't add performance and so is deleted
            if val[0] == "initial variable":
                x.drop(var, axis=1, inplace=True)

            else:

                # Conditions on imputation
                if val[0] == "most":
                    imputer = SimpleImputer(strategy="most_frequent")
                    imputer.fit(self.x_fitted[var].values.reshape(-1, 1))
                    x[var] = imputer.transform(x[var].values.reshape(-1, 1))
                    del val[0:2]
                
                elif val[0] == "nan":
                    x[var] = x[var].fillna('nan_category')
                    del val[0:2]

                # Conditions on encoding
                if val[0] == "one-hot":
                    encoder = OneHotEncoder(handle_unknown='ignore', sparse=False)
                    encoder.fit(self.x_fitted[var].values.reshape(-1, 1))
                    new_cols_names = encoder.get_feature_names_out(input_features=[var])
                    x[new_cols_names] = encoder.transform(x[var].values.reshape(-1, 1))
                    x.drop(var, axis=1, inplace=True)
                
                elif val[0] == "ordinal":
                    encoder = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=999) # A VOIR SI PERTINENT use_encoded_value
                    encoder.fit(self.x_fitted[var].values.reshape(-1, 1))
                    x["encoded_{}".format(var)] = encoder.transform(x[var].values.reshape(-1, 1))
                    x.drop(var, axis=1, inplace=True)

        return x

# ---------------------------------------------------------------------------------------------------------------
# Functions related to the class

def var_impute_encode_test(x_train, y_train, x_valid, y_valid, var, initial_scores, results_df, models_dict, ML_type, score, combination):
    
    x_train_bis = x_train.copy()
    x_valid_bis = x_valid.copy()

    # Proceed imputation on given variable
    if combination[0] == 'most_frequent':
        x_train_bis, x_valid_bis = apply_impute(x_train=x_train_bis, strategy="most_frequent", var=var, x_valid=x_valid_bis)
    
    elif combination[0] == 'nan_category':
        x_train_bis[var] = x_train_bis[var].fillna('nan_category')
        x_valid_bis[var] = x_valid_bis[var].fillna('nan_category')
        
    # Proceed encoding on given variable
    if combination[1] == 'one-hot':
        encoder = OneHotEncoder(handle_unknown='ignore', sparse=False)
        train_new_cols = encoder.fit_transform(x_train_bis[var].values.reshape(-1, 1))
        valid_new_cols = encoder.transform(x_valid_bis[var].values.reshape(-1, 1))
        new_cols_names = encoder.get_feature_names_out(input_features=[var])

        x_train_bis[new_cols_names] = train_new_cols
        x_valid_bis[new_cols_names] = valid_new_cols
        x_train_bis.drop(var, axis=1, inplace=True)
        x_valid_bis.drop(var, axis=1, inplace=True)
    
    elif combination[1] == 'ordinal':
        encoder = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=999) # A VOIR SI PERTINENT use_encoded_value
        x_train_bis[var] = encoder.fit_transform(x_train_bis[var].values.reshape(-1, 1))
        x_valid_bis[var] = encoder.transform(x_valid_bis[var].values.reshape(-1, 1))
        
    # We delete other categorical variables
    _, _, low_card_categorical_var, high_card_categorical_var = get_types(x_train_bis)
    categorical_var = low_card_categorical_var + high_card_categorical_var
    x_train_bis.drop(categorical_var, axis=1, inplace=True)
    x_valid_bis.drop(categorical_var, axis=1, inplace=True)

    # Computing and comparing results of imputation
    if ML_type == "Classification":
        new_scores = classifiers_test(x_train_bis, y_train, x_valid_bis, y_valid, models_dict=models_dict).loc[score]
    elif ML_type == "Regression":
        new_scores = regressors_test(x_train_bis, y_train, x_valid_bis, y_valid, models_dict=models_dict).loc[score]
    
    keep_transform = test_keep_transform(initial_scores, new_scores)

    # Saving results
    results_df["{}_{}_{}".format(combination[0], combination[1], var)] = [
        np.mean(new_scores), 
        (np.mean(new_scores) - np.mean(initial_scores)), 
        np.max(new_scores),
        (np.max(new_scores) - np.max(initial_scores)),
        ((np.mean(new_scores) - np.mean(initial_scores)) + (np.max(new_scores) - np.max(initial_scores))) / 2,
        keep_transform,
        "{}".format(var)
    ]
    
    return results_df

def apply_impute(x_train, strategy, var, x_valid=pd.DataFrame()):
    
    imputer = SimpleImputer(strategy=strategy)
    x_train[var] = imputer.fit_transform(x_train[var].values.reshape(-1, 1))
    x_valid[var] = imputer.transform(x_valid[var].values.reshape(-1, 1))
    
    return x_train, x_valid