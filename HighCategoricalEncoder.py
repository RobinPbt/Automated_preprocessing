from common import *

# ---------------------------------------------------------------------------------------------------------------
# HighCategoricalEncoder class

class HighCategoricalEncoder(Tester):
    """
    Class which apply different encodings on a x matrix and test these encodings on some models to infer the best encodings
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
    - encoding_list : endodings to test (list)
    
    Possible values are ['ordinal', 'embeddings']
    """
    
    def __init__(self, variables, score, models_dict, ML_type, train_size=0.8, random_state=None, encoding_list=['ordinal', 'embeddings']):
        self.encoding_list = encoding_list
        super().__init__(variables, score, models_dict, ML_type, train_size, random_state)
        
        # Test if imputation_list and encoding_list are correct
        possible_list = ['ordinal', 'embeddings']

        for encoding in self.encoding_list:
            if encoding not in possible_list:
                raise ValueError("Encoding name '{}' is incorrect, possible values are {}".format(encoding, possible_list))
        
    def fit(self, x, y):
        """
        Fitting HighCategoricalEncoder with x and y matrices to find best encodings. 
        Note that initial scores (base reference) corresponds to scores without categorical variables.
        """ 
        
        # Get variables types
        continuous_numerical_var, discrete_numerical_var, low_card_categorical_var, high_card_categorical_var = get_types(x)
        numerical_var = continuous_numerical_var + discrete_numerical_var
        
        # Split between train and validation set 
        x_train, x_valid, y_train, y_valid = train_test_split(x, y, train_size=self.train_size, random_state=self.random_state)
        
        # Define if the score must be maximized or minimized
        score_strategy = find_score_optmization_strategy(self.score)
        
        # DataFrame to store our results, we use only numerical values for initial scores
        self.results_df, initial_scores = init_test(
            x_train[numerical_var], y_train,
            x_valid[numerical_var], y_valid, 
            self.score, self.models_dict, self.ML_type, score_strategy
        )
        
        # We test each combination on each variable
        for var in self.variables: 
            for encoding in self.encoding_list:
                self.results_df = var_encode_test(
                    x_train, y_train, 
                    x_valid, y_valid, 
                    var, initial_scores, score_strategy, self.results_df, self.models_dict, self.ML_type, self.score, encode=encoding,
                )

        # We save best transformation for each variable
        self.best_results_df = pd.DataFrame(index=["Best result"])

        for var in self.variables: 
            self.best_results_df[var] = find_best_result(self.results_df, var, find_score_optmization_strategy(self.score))
        
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

            # We will force NaN category for imputation, see later if we add other options
            x[var] = x[var].fillna('nan_category')
            
            # First test if variable doesn't add performance and so is deleted
            if val[0] == "initial variable":
                x.drop(var, axis=1, inplace=True)

            else:
                # WE REFIT ALL MODEL WITH X, POSSIBLE IN ANOTHER WAY ?
                if val[0] == "embeddings":
                    
                    x_fitted_copy = self.x_fitted.copy()
                    x_fitted_copy[var] = x_fitted_copy[var].fillna('nan_category')

                    # Tokenize with SpaCy
                    nlp = spacy.load("en_core_web_sm")
                    docs = nlp.pipe(x_fitted_copy[var])

                    # Get lenght (nb tokens) of each doc and max
                    lenght_docs = [len(doc) for doc in docs]
                    max_length = max(lenght_docs)

                    # Define vocabulary size with a security margin of 20%
                    vocab_size = int(max(lenght_docs) + (0.2 * max(lenght_docs)))

                    # One-hot encode documents with keras and apply padding to get vector of same size 
                    padded_x_fitted = get_padding_vec(x_fitted_copy, var, vocab_size, max_length)
                    padded_x = get_padding_vec(x, var, vocab_size, max_length)

                    # Create our embedding model
                    model = embedding_model(vocab_size, max_length)

                    # Fit model with x_train --> EPOCHS IN PARAMETERS ?
                    model.fit(padded_x_fitted, self.y_fitted, epochs=100, verbose=0)

                    # Remove the output layer, to keep only the features when predicting
                    model = Model(inputs=model.inputs, outputs=model.layers[-2].output)

                    # Get embeddings
                    embeddings_x = model.predict(padded_x)

                    # Add it to matrices
                    x = concat_embeddings(embeddings_x, x, var)
                    x.drop(var, axis=1, inplace=True)
                    
                    
                elif val[0] == "ordinal":
                    encoder = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=999) # A VOIR SI PERTINENT use_encoded_value
                    encoder.fit(self.x_fitted[var].values.reshape(-1, 1))
                    x["encoded_{}".format(var)] = encoder.transform(x[var].values.reshape(-1, 1))
                    x.drop(var, axis=1, inplace=True)

        return x

# ---------------------------------------------------------------------------------------------------------------
# Functions related to the class

def get_padding_vec(x, var, vocab_size, max_length):
    
    encoded_x = [one_hot(text,vocab_size) for text in x[var]]
    padded_x = pad_sequences(encoded_x, maxlen=max_length, padding='post')
    
    return padded_x

def concat_embeddings(embeddings, x, var):
    
    emb_df = pd.DataFrame(
        embeddings, 
        index=x.index, 
        columns=["{}_embed{}".format(var, i+1) for i in range(embeddings.shape[1])],
    )
    
    x = pd.concat([x, emb_df], axis=1)
    
    return x

def embedding_model(vocab_size, max_length):
    
    # Create our embedding model --> FOR NOW WEIGHTS TRAINED ON BINARY CLASSIFICATION
    model = Sequential()
    embedding_layer = Embedding(input_dim=vocab_size, output_dim=64, input_length=max_length, mask_zero=True)
    model.add(embedding_layer)
    # conv_layer = Conv1D(16, 4, activation='relu')
    # model.add(conv_layer)
    model.add(GlobalMaxPooling1D())
    model.add(Dense(16))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['acc'])
    
    return model

def var_encode_test(x_train, y_train, x_valid, y_valid, var, initial_scores, score_strategy, results_df, models_dict, ML_type, score, encode):
    
    x_train_bis = x_train.copy()
    x_valid_bis = x_valid.copy()
    
    # We will force NaN category for imputation, see later if we add other options
    x_train_bis[var] = x_train_bis[var].fillna('nan_category')
    x_valid_bis[var] = x_valid_bis[var].fillna('nan_category')
    
    if encode == 'ordinal':
        encoder = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=999) # A VOIR SI PERTINENT use_encoded_value
        x_train_bis[var] = encoder.fit_transform(x_train_bis[var].values.reshape(-1, 1))
        x_valid_bis[var] = encoder.transform(x_valid_bis[var].values.reshape(-1, 1))
        
    elif encode == 'embeddings':
        
        # Tokenize with SpaCy
        nlp = spacy.load("en_core_web_sm")
        docs = nlp.pipe(x_train_bis[var])
        
        # Get lenght (nb tokens) of each doc and max
        lenght_docs = [len(doc) for doc in docs]
        max_length = max(lenght_docs)
        
        # Define vocabulary size with a security margin of 20%
        vocab_size = int(max(lenght_docs) + (0.2 * max(lenght_docs)))
        
        # One-hot encode documents with keras and apply padding to get vector of same size 
        padded_x_train = get_padding_vec(x_train_bis, var, vocab_size, max_length)
        padded_x_valid = get_padding_vec(x_valid_bis, var, vocab_size, max_length)
        
        # Create our embedding model
        model = embedding_model(vocab_size, max_length)
        
        # Fit model with x_train --> EPOCHS IN PARAMETERS ?
        model.fit(padded_x_train, y_train, epochs=100, verbose=0)
        
        # Remove the output layer, to keep only the features when predicting
        model = Model(inputs=model.inputs, outputs=model.layers[-2].output)
        
        # Get embeddings
        embeddings_x_train = model.predict(padded_x_train)
        embeddings_x_valid = model.predict(padded_x_valid)
        
        # Add it to matrices
        x_train_bis = concat_embeddings(embeddings_x_train, x_train_bis, var)
        x_valid_bis = concat_embeddings(embeddings_x_valid, x_valid_bis, var)
        x_train_bis.drop(var, axis=1, inplace=True)
        x_valid_bis.drop(var, axis=1, inplace=True)
        
        
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
    
    keep_transform = test_keep_transform(initial_scores, new_scores, score_strategy)

    # Saving results
    KPIs_scores = get_KPIs(initial_scores, new_scores, score_strategy, keep_transform, var)
    results_df["{}_{}".format(encode, var)] = KPIs_scores
    
    return results_df