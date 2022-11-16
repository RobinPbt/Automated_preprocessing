from common import *

# ---------------------------------------------------------------------------------------------------------------
# NumericalCombiner class

class NumericalCombiner(Tester):
    """
    Class which apply different numerical combinations between variables on a x matrix and test these imputations on some models to infer the best imputations
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
    - operation_list : operations to test (list)
    
    Possible values are ['sum', 'substraction', 'division', 'multiplication']
    """
    
    def __init__(self, variables, score, models_dict, ML_type, train_size=0.8, random_state=None, operation_list = ['sum', 'substraction', 'division', 'multiplication']):
        self.operation_list = operation_list
        super().__init__(variables, score, models_dict, ML_type, train_size, random_state)
        
        # Test if operation_list is correct
        possible_list = ['sum', 'substraction', 'division', 'multiplication']

        for operation in self.operation_list:
            if operation not in possible_list:
                raise ValueError("Operation name '{}' is incorrect, possible values are {}".format(operation, possible_list))

    def fit(self, x, y):
        """
        Fitting NumericalCombiner with x and y matrices to find best operations. 
        Note that initial scores (base reference) corresponds to scores without operations.
        """ 
        
        # Split between train and validation set 
        x_train, x_valid, y_train, y_valid = train_test_split(x, y, train_size=self.train_size, random_state=self.random_state)
        
        # DataFrame to store our results, here we also need to add the second variable of the sum
        self.results_df, initial_scores = init_test(x_train, y_train, x_valid, y_valid, self.score, self.models_dict, self.ML_type)
        row_add = pd.DataFrame(index=["Variable 2", "Operation"])
        row_add["initial_scores"] = np.nan
        self.results_df = pd.concat([self.results_df, row_add], axis=0)
        
        # We want to try every combination of 2 variables without redundant computations
        possible_combinations = itertools.combinations(self.variables, 2)
        
        # We test each operation on each variable
        for operation in self.operation_list:
            for var_1, var_2 in possible_combinations: 
                self.results_df = combined_features_test(
                    x_train, y_train, 
                    x_valid, y_valid, 
                    var_1, var_2, 
                    initial_scores, self.results_df, self.models_dict, self.ML_type, self.score,
                    operation=operation,
                )

        # We save all sums with positive impact on performance
        self.positive_results_df = find_positives_results(self.results_df)
        
        super().fit(x, y)
        
    def transform(self, x):
        """
        Transform a x matrix with best transformations found on fitting step. 
        These transformations are computed after the initial matrix used during fitting.
        """
        
        super().transform(x)
    
        for combination in self.positive_results_df.columns:
            var_1 = self.positive_results_df.loc['Initial variable', combination] 
            var_2 = self.positive_results_df.loc['Variable 2', combination]
            operation = self.positive_results_df.loc['Operation', combination]

            if operation == 'sum': # QUID COPY WARNING ?
                x[combination] = x.apply(lambda p: p[var_1] + p[var_2], axis=1)
            elif operation == 'substraction':
                x[combination] = x.apply(lambda p: p[var_1] - p[var_2], axis=1)
            elif operation == 'division':
                x[combination] = x.apply(lambda p: 0 if p[var_2] == 0 else p[var_1] / p[var_2], axis=1)
            elif operation == 'multiplication':
                x[combination] = x.apply(lambda p: p[var_1] * p[var_2], axis=1)

        return x

# ---------------------------------------------------------------------------------------------------------------
# Functions related to the class

def combined_features_test(x_train, y_train, x_valid, y_valid, var_1, var_2, initial_scores, results_df, models_dict, ML_type, score, operation):
    
    x_train_bis = x_train.copy()
    x_valid_bis = x_valid.copy()

    # Creating combined variable
    if operation == 'sum':
        x_train_bis["{}_{}_{}".format(var_1, operation, var_2)] = x_train_bis.apply(lambda p: p[var_1] + p[var_2], axis=1)
        x_valid_bis["{}_{}_{}".format(var_1, operation, var_2)] = x_valid_bis.apply(lambda p: p[var_1] + p[var_2], axis=1)
    elif operation == 'substraction':
        x_train_bis["{}_{}_{}".format(var_1, operation, var_2)] = x_train_bis.apply(lambda p: p[var_1] - p[var_2], axis=1)
        x_valid_bis["{}_{}_{}".format(var_1, operation, var_2)] = x_valid_bis.apply(lambda p: p[var_1] - p[var_2], axis=1)
    elif operation == 'division':
        x_train_bis["{}_{}_{}".format(var_1, operation, var_2)] = x_train_bis.apply(lambda p: 0 if p[var_2] == 0 else p[var_1] / p[var_2], axis=1)
        x_valid_bis["{}_{}_{}".format(var_1, operation, var_2)] = x_valid_bis.apply(lambda p: 0 if p[var_2] == 0 else p[var_1] / p[var_2], axis=1)
    elif operation == 'multiplication':
        x_train_bis["{}_{}_{}".format(var_1, operation, var_2)] = x_train_bis.apply(lambda p: p[var_1] * p[var_2], axis=1)
        x_valid_bis["{}_{}_{}".format(var_1, operation, var_2)] = x_valid_bis.apply(lambda p: p[var_1] * p[var_2], axis=1)
    
    # Computing and comparing results of imputation
    if ML_type == "Classification":
        new_scores = classifiers_test(x_train_bis, y_train, x_valid_bis, y_valid, models_dict=models_dict).loc[score]
    elif ML_type == "Regression":
        new_scores = regressors_test(x_train_bis, y_train, x_valid_bis, y_valid, models_dict=models_dict).loc[score]
    
    keep_transform = test_keep_transform(initial_scores, new_scores)

    # Saving results
    results_df["{}_{}_{}".format(var_1, operation, var_2)] = [
        np.mean(new_scores), 
        (np.mean(new_scores) - np.mean(initial_scores)), 
        np.max(new_scores),
        (np.max(new_scores) - np.max(initial_scores)),
        ((np.mean(new_scores) - np.mean(initial_scores)) + (np.max(new_scores) - np.max(initial_scores))) / 2,
        keep_transform,
        "{}".format(var_1),
        "{}".format(var_2),
        operation,
    ]
    
    return results_df