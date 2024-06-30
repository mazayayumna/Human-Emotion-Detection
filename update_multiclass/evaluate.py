import pandas as pd
import numpy as np
import pickle
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import load_model


from config import Config
from preprocessing import PreprocessText


def load_data(config):
    with open(config['tokenizer_path'], 'rb') as handle:
        tokenizer = pickle.load(handle)
    model = load_model(config['model_path'])
    return tokenizer, model

def main(config):
    # Load dataset
    emosi = pd.read_csv(config['dataset_path'])
    test_data = emosi[-config['test_size']:]

    # Preprocess text
    preprocessor = PreprocessText()
    instance = preprocessor(test_data)

    # Load tokenizer and model
    tokenizer, model = load_data(config)

    # tokenizing and padding
    instance = tokenizer.texts_to_sequences(instance['preprocessed'])
    x_test = pad_sequences(instance, padding='post', maxlen=config['max_length'])
    y_test = np.array(test_data['label'])
    score_test = model.evaluate(x_test, y_test, verbose=1)
    print(score_test)

if __name__ == "__main__":
    config = Config()
    main(config)
