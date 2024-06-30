import pandas as pd
import numpy as np
import re

class PreprocessText:
    """
    A class for preprocessing text data.

    Methods:
    - preprocess_text: Preprocesses a given sentence by converting it to lowercase, removing HTML tags, punctuations, numbers,
                       single characters, and multiple spaces.
    - __call__: Applies the preprocess_text method to a DataFrame column.

    Usage:
    preprocess = PreprocessText()
    preprocessed_data = preprocess(df)
    """

    @staticmethod
    def preprocess_text(sen: str) -> str:
        """
        Preprocesses a given sentence by converting it to lowercase, removing HTML tags, punctuations, numbers,
        single characters, and multiple spaces.

        Args:
        - sen: The input sentence to be preprocessed.

        Returns:
        - The preprocessed sentence.
        """
        #lowercase
        sentence = sen.lower()

        # Removing html tags
        TAG_RE = re.compile(r'<[^>]+>')
        sentence = TAG_RE.sub('', sentence)

        # Remove punctuations and numbers
        sentence = re.sub('[^a-zA-Z]', ' ', sentence)

        # Single character removal
        sentence = re.sub(r"\s+[a-zA-Z]\s+", ' ', sentence)

        # Removing multiple spaces
        sentence = re.sub(r'\s+', ' ', sentence)

        return sentence
    
    def __call__(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Applies the preprocess_text method to the 'text' column of a DataFrame.

        Args:
        - df: The input DataFrame.

        Returns:
        - The DataFrame with the 'text' column preprocessed.
        """
        data = df.copy()
        print('begin preprocess')
        data['preprocessed'] = data['text'].apply(lambda x: self.preprocess_text(x))
        return data