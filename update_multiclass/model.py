import numpy as np
from tensorflow.keras import Model, Sequential
from tensorflow.keras.layers import ( 
    Dense, 
    Embedding, 
    InputLayer, 
    GlobalMaxPooling1D,
    Conv1D)

class CNNModel:
    """
    A class representing a Convolutional Neural Network (CNN) model for human emotion detection.

    Attributes:
        max_len (int): The maximum length of input sequences.
        embedding_matrix (np.array): The pre-trained embedding matrix.
        loss (str): The loss function used for training the model.
        optimizer (str): The optimizer used for training the model.
        metrics (list): The evaluation metrics used for training the model.
        activation (str): The activation function used in the output layer.
        num_classes (int): The number of classes for emotion detection.

    Methods:
        build_model(): Builds and compiles the CNN model.
    """

    def __init__(self, max_len: int, embedding_matrix: np.array, loss: str, optimizer: str, activation: str, num_classes: int):
        """
        Initializes a CNNModel object.

        Args:
            max_len (int): The maximum length of input sequences.
            embedding_matrix (np.array): The pre-trained embedding matrix.
            loss (str): The loss function used for training the model.
            optimizer (str): The optimizer used for training the model.
            activation (str): The activation function used in the output layer.
            num_classes (int): The number of classes for emotion detection.
        """
        self.max_len = max_len
        self.embedding_matrix = embedding_matrix
        self.loss = loss
        self.optimizer = optimizer
        self.activation = activation
        self.num_classes = num_classes
        
    def build_model(self) -> Model:
        """
        Builds and compiles the CNN model.

        Returns:
            Model: The compiled CNN model.
        """
        model = Sequential()
        embedding_layer = Embedding(
            input_dim = self.embedding_matrix.shape[0], 
            output_dim = self.embedding_matrix.shape[1], 
            weights=[self.embedding_matrix], 
            input_length=self.max_len, 
            trainable=True)
        model.add(embedding_layer)
        model.add(Conv1D(filters=128, kernel_size=5, activation='relu'))
        model.add(GlobalMaxPooling1D())
        model.add(Dense(self.num_classes, activation=self.activation))
        model.build(input_shape=(None, self.max_len))  # Change the number of units to 6 and use softmax activation
        model.compile(optimizer=self.optimizer, loss=self.loss, metrics=['accuracy'])  # Change the loss function to categorical_crossentropy
        return model
    
    def __call__(self):
        """
        Calls the build_model method to create and return the CNN model.

        Returns:
            Model: The compiled CNN model.
        """
        print('build model')
        model = self.build_model()
        return model