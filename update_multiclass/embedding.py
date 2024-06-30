import numpy as np
from tensorflow.keras.preprocessing.text import Tokenizer

class EmbeddingModel:
    """
    A class that represents an embedding model.

    Attributes:
        max_len (int): The maximum length of the input sequence.
        embedding_path (str): The path to the file containing the word embeddings.
        tokenizer (Tokenizer): A tokenizer object for tokenizing text data.

    Methods:
        call_embedding: Loads the word embeddings from the specified file.
        create_embedding_matrix: Creates an embedding matrix for the vocabulary.
    """

    def __init__(self, max_len: int, embedding_path: str, tokenizer: Tokenizer):
        """
        Initializes an instance of the EmbeddingModel class.

        Args:
            max_len (int): The maximum length of the input sequence.
            embedding_path (str): The path to the file containing the word embeddings.
        """
        self.vocab_size = len(tokenizer.word_index) + 1
        self.max_len = max_len
        self.embedding_path = embedding_path
        self.tokenizer = tokenizer
    
    def call_embedding(self) -> dict:
        """
        Loads the word embeddings from the specified file.

        Returns:
            dict: A dictionary mapping words to their corresponding embedding vectors.
        """
        embeddings_dictionary = dict()
        with open(self.embedding_path, encoding="utf8") as glove_file:
            for line in glove_file:
                records = line.split()
                word = records[0]
                vector_dimensions = np.asarray(records[1:], dtype='float32')
                embeddings_dictionary[word] = vector_dimensions
        return embeddings_dictionary
    
    def create_embedding_matrix(self, embeddings_dictionary: dict) -> np.array:
        """
        Creates an embedding matrix for the vocabulary.

        Args:
            embeddings_dictionary (dict): A dictionary mapping words to their corresponding embedding vectors.

        Returns:
            np.array: An embedding matrix of shape (vocab_size, embedding_dim).
        """
        embedding_matrix = np.zeros((self.vocab_size, 100))
        for word, index in self.tokenizer.word_index.items():
            embedding_vector = embeddings_dictionary.get(word)
            if embedding_vector is not None:
                embedding_matrix[index] = embedding_vector
        return embedding_matrix
    
    def __call__(self):
        """
        Calls the embedding model.

        Returns:
            np.array: An embedding matrix of shape (vocab_size, embedding_dim).
        """
        print('begin embedding')
        embeddings_dictionary = self.call_embedding()
        embedding_matrix = self.create_embedding_matrix(embeddings_dictionary)
        return embedding_matrix

