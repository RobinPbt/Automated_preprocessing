from common import *
from NumericalImputer import NumericalImputer
from CategoricalImputerEncoder import CategoricalImputerEncoder
from HighCategoricalEncoder import HighCategoricalEncoder
from NumericalTransformer import NumericalTransformer
from NumericalCombiner import NumericalCombiner

def full_x_processing(
    x, 
    y, 
    score,
    models_dict,
    random_state=None, 
    continuous_numerical_var=None, 
    discrete_numerical_var=None, 
    low_card_categorical_var=None, 
    high_card_categorical_var=None,
    nb_iter_transformer=1,
    nb_iter_combinations=1,
):
    
    # Get variables types
    if not continuous_numerical_var:
        continuous_numerical_var, _, _, _ = get_types(x)
    if not discrete_numerical_var:
        _, discrete_numerical_var, _, _ = get_types(x)
    if not low_card_categorical_var:
        _, _, low_card_categorical_var, _ = get_types(x)
    if not high_card_categorical_var:
        _, _, _, high_card_categorical_var = get_types(x)
    
    numerical_var = continuous_numerical_var + discrete_numerical_var
    categorical_var = low_card_categorical_var + high_card_categorical_var
    
    x_transformed = x.copy()
    
    # Find and apply best numerical imputations
    imputer = NumericalImputer(variables=numerical_var, score=score, models_dict=models_dict, random_state=random_state)
    x_transformed = imputer.fit_transform(x_transformed, y)
    print("Imputation done : 1/5")
    
    # Find best categorical encoding and imputation
    encoder = CategoricalImputerEncoder(variables=low_card_categorical_var, score=score, models_dict=models_dict, random_state=random_state)
    x_transformed = encoder.fit_transform(x_transformed, y)
    print("Low cardinality encoding done : 2/5")
    
    # Find best categorical encoding for high cardinality variables
    high_encoder = HighCategoricalEncoder(variables=high_card_categorical_var, score=score, models_dict=models_dict, random_state=random_state)
    x_transformed = high_encoder.fit_transform(x_transformed, y)
    print("High cardinality encoding done : 3/5")
    
    # Find best numerical transformations
    dict_transformers = {}
    
    for i in range(nb_iter_transformer):
        transformer = NumericalTransformer(variables=continuous_numerical_var, score=score, models_dict=models_dict, random_state=random_state)
        x_transformed = transformer.fit_transform(x_transformed, y)
        continuous_numerical_var = transformer.variables_transformed
        numerical_var = continuous_numerical_var + discrete_numerical_var
        dict_transformers["transformer_{}".format(i+1)] = transformer
    print("Transformation done : 4/5")
    
    # Find combined features improving performance
    dict_combiners = {}
    
    for i in range(nb_iter_combinations):
        combiner = NumericalCombiner(variables=numerical_var, score=score, models_dict=models_dict, random_state=random_state)
        x_transformed = combiner.fit_transform(x_transformed, y)
        dict_transformers["combiner_{}".format(i+1)] = combiner
    print("Combination done : 5/5")
    
    return x_transformed, imputer, encoder, high_encoder, dict_transformers, dict_combiners

def full_x_test_processing(x_test, imputer, encoder, high_encoder, dict_transformers, dict_combiners):
    
    x_test_transformed = x_test.copy()
    
    x_test_transformed = imputer.transform(x_test_transformed)
    x_test_transformed = encoder.transform(x_test_transformed)
    x_test_transformed = high_encoder.transform(x_test_transformed)
    
    for transformer in dict_transformers.values():
        x_test_transformed = transformer.transform(x_test_transformed)
    
    for combiner in dict_combiners.values(): 
        x_test_transformed = combiner.transform(x_test_transformed)
    
    return x_test_transformed