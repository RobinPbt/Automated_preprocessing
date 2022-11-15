from common import *

# ---------------------------------------------------------------------------------------------------------------
# NumericalTransformer class

class NumericalTransformer(Tester):
    """
    Class which apply different numerical transformations on a x matrix and test these imputations on some models to infer the best imputations
    given models scores. The scoring is performed on a validation set created from the fitted matrice."" 
    
    Parameters
    ----------
    - variables : name of variables to be processed, must be numerical variables (list)
    - score : score to be optimized (XX)
    - random_state : RandomState instance (int)
    - train_size : percentage of the dataset used as a training set (float in range [0:1]) 
    - transformation_list : transformations to test (list)
    
    Possible values are ['log', 'discretize', 'standardize', 'normalize', 'power', 'robust_scale', 'square', 'sqrt', 'cos', 'tan', 'sin']
    """
    
    def __init__(self, variables, score, models_dict, train_size=0.8, random_state=None, transformation_list=['log', 'discretize', 'standardize', 'normalize', 'power', 'robust_scale', 'square', 'sqrt', 'cos', 'tan', 'sin']):
        self.transformation_list = transformation_list
        super().__init__(variables, score, models_dict, train_size, random_state)
        self.variables_transformed = variables.copy()
        
        # Test if transformation_list is correct
        possible_list = ['log', 'discretize', 'standardize', 'normalize', 'power', 'robust_scale', 'square', 'sqrt', 'cos', 'tan', 'sin']

        for transformation in self.transformation_list:
            if transformation not in possible_list:
                raise ValueError("Transformation name '{}' is incorrect, possible values are {}".format(transformation, possible_list))
        
    def fit(self, x, y):
        """
        Fitting NumericalTransformer with x and y matrices to find best transformations. 
        Note that initial scores (base reference) corresponds to scores without transformation.
        """ 
        
        # Split between train and validation set 
        x_train, x_valid, y_train, y_valid = train_test_split(x, y, train_size=self.train_size, random_state=self.random_state)
        
        # DataFrame to store our results
        self.results_df, initial_scores = init_test(x_train, y_train, x_valid, y_valid, self.score, self.models_dict)        
        
        # We test each transformation on each variable
        for var in self.variables: 
            for transformation in self.transformation_list:
                self.results_df = var_transform_test(
                    x_train, y_train,
                    x_valid, y_valid,
                    var, initial_scores, self.results_df, self.models_dict, self.score, self.random_state, transform=transformation,
                )

        # We save best transformation for each variable
        self.best_results_df = pd.DataFrame(index=["Best result"])

        for var in self.variables: 
            self.best_results_df[var] = find_best_result(self.results_df, var)
        
        super().fit(x, y)
        
    def transform(self, x):
        """
        Transform a x matrix with best transformations found on fitting step. 
        These transformations are computed after the initial matrix used during fitting.
        """
        
        super().transform(x)
    
        for var in self.best_results_df.columns:
            val = self.best_results_df[var].values[0]
            val = val.split("_")

            # First test if variable doesn't add performance and in that case no action
            if val[0] == "initial variable":
                pass

            else:

                if val[0] == "log":
                    x["log_{}".format(var)] = np.log(x[var] + 1)
                    x = nan_inf_replace(x, "log_{}".format(var))
                    x.drop(var, axis=1, inplace=True)
                    if var in self.variables_transformed:
                        idx = self.variables_transformed.index(var)
                        self.variables_transformed[idx] = "log_{}".format(var)
                
                elif val[0] == "discretize":
                    discretizer = KBinsDiscretizer(n_bins=5, encode='ordinal', strategy='quantile', random_state=self.random_state)
                    discretizer.fit(self.x_fitted[var].values.reshape(-1, 1))
                    x["discretized_{}".format(var)] = discretizer.transform(x[var].values.reshape(-1, 1))
                    x = nan_inf_replace(x, "discretized_{}".format(var))
                    x.drop(var, axis=1, inplace=True)
                    if var in self.variables_transformed:
                        idx = self.variables_transformed.index(var)
                        self.variables_transformed[idx] = "discretized_{}".format(var)
                    
                elif val[0] == "standardize":
                    std_scaler = StandardScaler()
                    std_scaler.fit(self.x_fitted[var].values.reshape(-1, 1))
                    x["standardized_{}".format(var)] = std_scaler.transform(x[var].values.reshape(-1, 1))
                    x = nan_inf_replace(x, "standardized_{}".format(var))
                    x.drop(var, axis=1, inplace=True)
                    if var in self.variables_transformed:
                        idx = self.variables_transformed.index(var)
                        self.variables_transformed[idx] = "standardized_{}".format(var)

                elif val[0] == "normalize":
                    normalizer = Normalizer()
                    normalizer.fit(self.x_fitted[var].values.reshape(-1, 1))
                    x["normalized_{}".format(var)] = normalizer.transform(x[var].values.reshape(-1, 1))
                    x = nan_inf_replace(x, "normalized_{}".format(var))
                    x.drop(var, axis=1, inplace=True)
                    if var in self.variables_transformed:
                        idx = self.variables_transformed.index(var)
                        self.variables_transformed[idx] = "normalized_{}".format(var)

                elif val[0] == "power":
                    power = PowerTransformer()
                    power.fit(self.x_fitted[var].values.reshape(-1, 1))
                    x["power_{}".format(var)] = power.transform(x[var].values.reshape(-1, 1))
                    x = nan_inf_replace(x, "power_{}".format(var))
                    x.drop(var, axis=1, inplace=True)
                    if var in self.variables_transformed:
                        idx = self.variables_transformed.index(var)
                        self.variables_transformed[idx] = "power_{}".format(var)

                elif val[0] == "robust":
                    robust_scaler = RobustScaler()
                    robust_scaler.fit(self.x_fitted[var].values.reshape(-1, 1))
                    x["robust_{}".format(var)] = robust_scaler.transform(x[var].values.reshape(-1, 1))
                    x = nan_inf_replace(x, "robust_{}".format(var))
                    x.drop(var, axis=1, inplace=True)
                    if var in self.variables_transformed:
                        idx = self.variables_transformed.index(var)
                        self.variables_transformed[idx] = "robust_{}".format(var)
                    
                elif val[0] == "square":
                    x["square_{}".format(var)] = x[var].apply(np.square)
                    x = nan_inf_replace(x, "square_{}".format(var))
                    x.drop(var, axis=1, inplace=True)
                    if var in self.variables_transformed:
                        idx = self.variables_transformed.index(var)
                        self.variables_transformed[idx] = "square_{}".format(var)

                elif val[0] == "sqrt":
                    x["sqrt_{}".format(var)] = x[var].apply(np.sqrt)
                    x = nan_inf_replace(x, "sqrt_{}".format(var))
                    x.drop(var, axis=1, inplace=True)
                    if var in self.variables_transformed:
                        idx = self.variables_transformed.index(var)
                        self.variables_transformed[idx] = "sqrt_{}".format(var)

                elif val[0] == "cos":
                    x["cos_{}".format(var)] = x[var].apply(np.cos)
                    x = nan_inf_replace(x, "cos_{}".format(var))
                    x.drop(var, axis=1, inplace=True)
                    if var in self.variables_transformed:
                        idx = self.variables_transformed.index(var)
                        self.variables_transformed[idx] = "cos_{}".format(var)

                elif val[0] == "tan":
                    x["tan_{}".format(var)] = x[var].apply(np.tan)
                    x = nan_inf_replace(x, "tan_{}".format(var))
                    x.drop(var, axis=1, inplace=True)
                    if var in self.variables_transformed:
                        idx = self.variables_transformed.index(var)
                        self.variables_transformed[idx] = "tan_{}".format(var)

                elif val[0] == "sin":
                    x["sin_{}".format(var)] = x[var].apply(np.sin)
                    x = nan_inf_replace(x, "sin_{}".format(var))
                    x.drop(var, axis=1, inplace=True)
                    if var in self.variables_transformed:
                        idx = self.variables_transformed.index(var)
                        self.variables_transformed[idx] = "sin_{}".format(var)

        return x

# ---------------------------------------------------------------------------------------------------------------
# Functions related to the class

def log_var(x, var):
    
    x["log_{}".format(var)] = np.log(x[var] + 1)
    x.drop(var, axis=1, inplace=True)
    
    return x

def var_transform_test(x_train, y_train, x_valid, y_valid, var, initial_scores, results_df, models_dict, score, random_state, transform):
    
    x_train_bis = x_train.copy()
    x_valid_bis = x_valid.copy()

    # Creating transformed variable to replace initial one
    if transform == "log":
        x_train_bis = log_var(x_train_bis, var)
        x_train_bis = nan_inf_replace(x_train_bis, "log_{}".format(var))
        
        x_valid_bis = log_var(x_valid_bis, var)
        x_valid_bis = nan_inf_replace(x_valid_bis, "log_{}".format(var))
        
    elif transform == "discretize":
        discretizer = KBinsDiscretizer(n_bins=5, encode='ordinal', strategy='quantile', random_state=random_state)
        x_train_bis[var] = discretizer.fit_transform(x_train_bis[var].values.reshape(-1, 1))
        x_train_bis = nan_inf_replace(x_train_bis, var)
        
        x_valid_bis[var] = discretizer.transform(x_valid_bis[var].values.reshape(-1, 1))
        x_valid_bis = nan_inf_replace(x_valid_bis, var)
    
    elif transform == "standardize":
        std_scaler = StandardScaler()
        x_train_bis[var] = std_scaler.fit_transform(x_train_bis[var].values.reshape(-1, 1))
        x_train_bis = nan_inf_replace(x_train_bis, var)
        
        x_valid_bis[var] = std_scaler.transform(x_valid_bis[var].values.reshape(-1, 1))
        x_valid_bis = nan_inf_replace(x_valid_bis, var)
        
    elif transform == "normalize":
        normalizer = Normalizer()
        x_train_bis[var] = normalizer.fit_transform(x_train_bis[var].values.reshape(-1, 1))
        x_train_bis = nan_inf_replace(x_train_bis, var)
        
        x_valid_bis[var] = normalizer.transform(x_valid_bis[var].values.reshape(-1, 1))
        x_valid_bis = nan_inf_replace(x_valid_bis, var)
        
    elif transform == "power":
        power = PowerTransformer()
        x_train_bis[var] = power.fit_transform(x_train_bis[var].values.reshape(-1, 1))
        x_train_bis = nan_inf_replace(x_train_bis, var)
        
        x_valid_bis[var] = power.transform(x_valid_bis[var].values.reshape(-1, 1))
        x_valid_bis = nan_inf_replace(x_valid_bis, var)
        
    elif transform == "robust_scale":
        robust_scaler = RobustScaler()
        x_train_bis[var] = robust_scaler.fit_transform(x_train_bis[var].values.reshape(-1, 1))
        x_train_bis = nan_inf_replace(x_train_bis, var)
        
        x_valid_bis[var] = robust_scaler.transform(x_valid_bis[var].values.reshape(-1, 1))
        x_valid_bis = nan_inf_replace(x_valid_bis, var)
        
    elif transform == "square":
        x_train_bis[var] = x_train_bis[var].apply(np.square)
        x_train_bis = nan_inf_replace(x_train_bis, var)
        
        x_valid_bis[var] = x_valid_bis[var].apply(np.square)
        x_valid_bis = nan_inf_replace(x_valid_bis, var)
    
    elif transform == "sqrt":
        x_train_bis[var] = x_train_bis[var].apply(np.sqrt)
        x_train_bis = nan_inf_replace(x_train_bis, var)
        
        x_valid_bis[var] = x_valid_bis[var].apply(np.sqrt)
        x_valid_bis = nan_inf_replace(x_valid_bis, var)
        
    elif transform == "cos":
        x_train_bis[var] = x_train_bis[var].apply(np.cos)
        x_train_bis = nan_inf_replace(x_train_bis, var)
        
        x_valid_bis[var] = x_valid_bis[var].apply(np.cos)
        x_valid_bis = nan_inf_replace(x_valid_bis, var)
        
    elif transform == "tan":
        x_train_bis[var] = x_train_bis[var].apply(np.tan)
        x_train_bis = nan_inf_replace(x_train_bis, var)
        
        x_valid_bis[var] = x_valid_bis[var].apply(np.tan)
        x_valid_bis = nan_inf_replace(x_valid_bis, var)
        
    elif transform == "sin":
        x_train_bis[var] = x_train_bis[var].apply(np.sin)
        x_train_bis = nan_inf_replace(x_train_bis, var)
        
        x_valid_bis[var] = x_valid_bis[var].apply(np.sin)
        x_valid_bis = nan_inf_replace(x_valid_bis, var)
        
    # Computing and comparing results of transformation 
    new_scores = classifiers_test(x_train_bis, y_train, x_valid_bis, y_valid, models_dict=models_dict).loc[score]
    keep_transform = test_keep_transform(initial_scores, new_scores)

    # Saving results
    results_df["{}_{}".format(transform, var)] = [
        np.mean(new_scores), 
        (np.mean(new_scores) - np.mean(initial_scores)), 
        np.max(new_scores),
        (np.max(new_scores) - np.max(initial_scores)),
        ((np.mean(new_scores) - np.mean(initial_scores)) + (np.max(new_scores) - np.max(initial_scores))) / 2,
        keep_transform,
        "{}".format(var)
    ]
    
    return results_df