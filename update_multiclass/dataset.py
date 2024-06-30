import pandas as pd
import numpy as np
import pickle

from tensorflow.keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences

class CreateDataset:
    """
    A class to create a dataset for human emotion detection.

    Args:
        max_features (int): The maximum number of words to keep based on word frequency.
        max_len (int): The maximum length of the sequences.
        train_size (int): The size of the training dataset.
        random_state (int): The random seed for reproducibility.

    Returns:
        tuple: A tuple containing the input features (X) and the corresponding labels (y).
    """

    def __init__(self, max_features: int, max_len: int, test_size: int, random_state: int, tokenizer_path: str):
        self.max_features = max_features
        self.max_len = max_len
        self.test_size = test_size
        self.random_state = random_state
        self.tokenizer_path = tokenizer_path
            
    def __call__(self, df: pd.DataFrame ) -> tuple:
        """
        Create a dataset based on the given DataFrame.

        Args:
            df (pd.DataFrame): The DataFrame containing the text and label columns.

        Returns:
            tokenizer: A tokenizer object for tokenizing text data.
            tuple: A tuple containing the input features (X) and the corresponding labels (y)
        """

        data = df[:-self.test_size]
        Y_train = data['label']

        tokenizer = Tokenizer(num_words=self.max_features)
        tokenizer.fit_on_texts(data['preprocessed'])

        X_train = tokenizer.texts_to_sequences(data['preprocessed'])
        X_train = pad_sequences(X_train, maxlen=self.max_len, padding='post')

        # saving tokenizer
        with open(self.tokenizer_path, 'wb') as handle:
            pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)
        
        return tokenizer, np.array(X_train), np.array(Y_train)