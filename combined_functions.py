from common import *
from NumericalImputer import NumericalImputer
from CategoricalImputerEncoder import CategoricalImputerEncoder
from HighCategoricalEncoder import HighCategoricalEncoder
from NumericalTransformer import NumericalTransformer
from NumericalCombiner import NumericalCombiner
from FeatureSelector import FeatureSelector

class FullPreprocessor(Tester):
    
    """
    Class which apply preprocessing steps according to specified pipeline.
    
    Parameters
    ----------
    
    - score : score to be optimized (XX)
    - models_dict : dict of models to test with name as key and model as value, 
    ex : {"Random Forest" : RandomForestRegressor(random_state=5)}. Note that only wrappers methods use these models
    - ML_type : type of machine learning algorithm, can be 'Regression' or 'Classification' (str)
    - random_state : RandomState instance (int)
    - continuous_numerical_var : variables considered numrical continuous, automatically infered if value set to 'None' (list or None)
    - discrete_numerical_var : variables considered numrical discrete, automatically infered if value set to 'None' (list or None)
    - low_card_categorical_var : variables considered categorical with low number of categories, automatically infered if value set to 'None' (list or None)
    - high_card_categorical_var : variables considered categorical with high/infinite number of categories, automatically infered if value set to 'None' (list or None)
    - nb_iter_transformer : number of iterations with NumericalTransformer, increasing the number creates mor complex transformations (int in range [1:inf])
    - nb_iter_combinations : number of iterations with NumericalCombiner, increasing the number creates more complex transformations (int in range [1:inf])
    - pipeline : preprocessing steps (dict)
    
    Possible methods_list values are :
    {
    'NumericalImputer' : {'imputation_list' : ['mean', 'median', 'most_frequent', 'constant']},
    'CategoricalImputerEncoder' : {'imputation_list' : ['most_frequent', 'nan_category'], 'encoding_list' : ['one-hot', 'ordinal']},
    'HighCategoricalEncoder' : {'encoding_list' : ['ordinal', 'embeddings']},
    'NumericalTransformer' : {'transformation_list' : ['log', 'discretize', 'standardize', 'normalize', 'power', 'robust_scale', 'square', 'sqrt', 'cos', 'tan', 'sin']},
    'NumericalCombiner' : {'operation_list' : ['sum', 'substraction', 'division', 'multiplication']},
    'FeatureSelector' : {'methods_list' : ['Variance Threshold', 'VIF Threshold', 'Backward Selection', 'Forward Selection', 'Recursive Feature Elimination',
    'Filter Pearson Coefficient', 'Filter Spearman Rho', 'Filter Kendall Tau', 'Filter Mutual information reg',
    'Filter Point-biserial correlation', 'Filter Chi2', 'Filter ANOVA F-score', 'Filter Mutual information clas']},
    }
    
    Please note that for FeatureSelector you should choose relevants methods for your machine learning problem (regression or classification). 
    See FeatureSelector documentation for more details.
    
    Please also note that most machine learning algorithms don't accept NaN values and non-numerical values, 
    thus you must specify at least one imputation method of each imputation class (NumericalImputer, CategoricalImputerEncoder) 
    and at least one encoding method for each encoding class (CategoricalImputerEncoder and HighCategoricalEncoder)
    if you want to use other classes (NumericalTransformer, NumericalCombiner and FeatureSelector)
    """

    def __init__(self, score, models_dict, ML_type, random_state=None,
                 continuous_numerical_var=None, discrete_numerical_var=None, low_card_categorical_var=None, high_card_categorical_var=None,
                 nb_iter_transformer=1, nb_iter_combinations=1, pipeline=None):     
        
        self.score = score
        self.models_dict = models_dict
        self.ML_type = ML_type
        self.random_state = random_state
        self.continuous_numerical_var = continuous_numerical_var
        self.discrete_numerical_var = discrete_numerical_var
        self.low_card_categorical_var = low_card_categorical_var
        self.high_card_categorical_var = high_card_categorical_var
        self.nb_iter_transformer = nb_iter_transformer
        self.nb_iter_combinations = nb_iter_combinations
        self.pipeline = pipeline
        
        self.imputer = NumericalImputer(None, None, None, None)
        self.encoder = CategoricalImputerEncoder(None, None, None, None)
        self.high_encoder = HighCategoricalEncoder(None, None, None, None)
        self.dict_transformers = {'NumericalTransformer' : NumericalTransformer([], None, None, None)}
        self.dict_combiners = {'NumericalCombiner' : NumericalCombiner(None, None, None, None)}
        self.selector = FeatureSelector(None, None, [])
        
        self.is_fitted = False
        self.x_fitted = None
        self.y_fitted = None
        
        # If no pipeline specified we proceed to all possible preprocessing
        if not self.pipeline:
            self.pipeline = {
                'NumericalImputer' : {'imputation_list' : ['mean', 'median', 'most_frequent', 'constant']},
                'CategoricalImputerEncoder' : {'imputation_list' : ['most_frequent', 'nan_category'], 'encoding_list' : ['one-hot', 'ordinal']},
                'HighCategoricalEncoder' : {'encoding_list' : ['ordinal', 'embeddings']},
                'NumericalTransformer' : {'transformation_list' : ['log', 'discretize', 'standardize', 'normalize', 'power', 'robust_scale', 'square', 'sqrt', 'cos', 'tan', 'sin']},
                'NumericalCombiner' : {'operation_list' : ['sum', 'substraction', 'division', 'multiplication']},
                'FeatureSelector' : {'methods_list' : [
                    'Variance Threshold', 
                    'VIF Threshold', 
                    'Backward Selection', 
                    'Forward Selection', 
                    'Recursive Feature Elimination',
                    'Filter Pearson Coefficient', 
                    'Filter Spearman Rho', 
                    'Filter Kendall Tau', 
                    'Filter Mutual information reg',
                ] if ML_type == 'Regression' else [
                    'Variance Threshold', 
                    'VIF Threshold', 
                    'Backward Selection', 
                    'Forward Selection', 
                    'Recursive Feature Elimination',
                    'Filter Point-biserial correlation', 
                    'Filter Chi2', 
                    'Filter ANOVA F-score', 
                    'Filter Mutual information clas'
                ]},
            }
             
    def fit(self, x, y):
        
        x_transformed = x.copy()
        
        # If variables types aren't specified we automatically infer them
        if not self.continuous_numerical_var:
            self.continuous_numerical_var, _, _, _ = get_types(x)
        if not self.discrete_numerical_var:
            _, self.discrete_numerical_var, _, _ = get_types(x)
        if not self.low_card_categorical_var:
            _, _, self.low_card_categorical_var, _ = get_types(x)
        if not self.high_card_categorical_var:
            _, _, _, self.high_card_categorical_var = get_types(x)

        numerical_var = self.continuous_numerical_var + self.discrete_numerical_var
        categorical_var = self.low_card_categorical_var + self.high_card_categorical_var
        
        # Find and apply best numerical imputations with NumericalImputer
        if 'NumericalImputer' in self.pipeline.keys():
            self.imputer = NumericalImputer(
                variables=numerical_var, score=self.score, models_dict=self.models_dict, 
                ML_type=self.ML_type, random_state=self.random_state, 
                imputation_list=self.pipeline['NumericalImputer']['imputation_list']
            )
            x_transformed = self.imputer.fit_transform(x_transformed, y)
            print("Imputation done")

        # Find best categorical encoding and imputation with CategoricalImputerEncoder
        if 'CategoricalImputerEncoder' in self.pipeline.keys():
            self.encoder = CategoricalImputerEncoder(
                variables=self.low_card_categorical_var, score=self.score, models_dict=self.models_dict, 
                ML_type=self.ML_type, random_state=self.random_state, 
                imputation_list=self.pipeline['CategoricalImputerEncoder']['imputation_list'], encoding_list=self.pipeline['CategoricalImputerEncoder']['encoding_list']
            )
            x_transformed = self.encoder.fit_transform(x_transformed, y)
            print("Low cardinality encoding done")

        # Find best categorical encoding for high cardinality variables with HighCategoricalEncoder
        if 'HighCategoricalEncoder' in self.pipeline.keys():
            self.high_encoder = HighCategoricalEncoder(
                variables=self.high_card_categorical_var, score=self.score, models_dict=self.models_dict, 
                ML_type=self.ML_type, random_state=self.random_state,
                encoding_list=self.pipeline['HighCategoricalEncoder']['encoding_list'],
            )
            x_transformed = self.high_encoder.fit_transform(x_transformed, y)
            print("High cardinality encoding done")

        # Find best numerical transformations with NumericalTransformer
        self.dict_transformers = {}

        if 'NumericalTransformer' in self.pipeline.keys():
            for i in range(self.nb_iter_transformer): # We can operate successive transformations on variables
                transformer = NumericalTransformer(
                    variables=self.continuous_numerical_var, score=self.score, models_dict=self.models_dict, 
                    ML_type=self.ML_type, random_state=self.random_state,
                    transformation_list=self.pipeline['NumericalTransformer']['transformation_list'],  
                )
                x_transformed = transformer.fit_transform(x_transformed, y)
                self.continuous_numerical_var = transformer.variables_transformed
                numerical_var = self.continuous_numerical_var + self.discrete_numerical_var
                self.dict_transformers["transformer_{}".format(i+1)] = transformer
            print("Transformation done")

        # Find combined features improving performance with NumericalCombiner
        self.dict_combiners = {}

        if 'NumericalCombiner' in self.pipeline.keys():
            for i in range(self.nb_iter_combinations): # We can operate successive combinations on variables
                combiner = NumericalCombiner(
                    variables=numerical_var, score=self.score, models_dict=self.models_dict, 
                    ML_type=self.ML_type, random_state=self.random_state,
                    operation_list=self.pipeline['NumericalCombiner']['operation_list'],
                )
                x_transformed = combiner.fit_transform(x_transformed, y)
                self.dict_combiners["combiner_{}".format(i+1)] = combiner
            print("Combination done")
        
        # Find best subset of variables with FeatureSelector
        if 'FeatureSelector' in self.pipeline.keys():
            self.selector = FeatureSelector(models_dict=self.models_dict, ML_type=self.ML_type, methods_list=self.pipeline['FeatureSelector']['methods_list'])
            x_transformed = self.selector.fit_transform(x_transformed, y)
            print("Feature selection done")
                      
        super().fit(x, y)
            
    def transform(self, x):
        
        super().transform(x)
        
        x_transformed = x.copy()
        
        if self.imputer.is_fitted:
            x_transformed = self.imputer.transform(x_transformed)
        
        if self.encoder.is_fitted:
            x_transformed = self.encoder.transform(x_transformed)
        
        if self.high_encoder.is_fitted:
            x_transformed = self.high_encoder.transform(x_transformed)

        for transformer in self.dict_transformers.values():
            if transformer.is_fitted:
                x_transformed = transformer.transform(x_transformed)

        for combiner in self.dict_combiners.values():
            if combiner.is_fitted:
                x_transformed = combiner.transform(x_transformed)
                
        if self.selector.is_fitted:
            x_transformed = self.selector.transform(x_transformed)

        return x_transformed