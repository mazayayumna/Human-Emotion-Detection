import pickle
import yaml
import pandas as pd
import numpy as np
import pprint
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import load_model

from config import Config
from preprocessing import PreprocessText


def load_data(config):
    with open(config['tokenizer_path'], 'rb') as handle:
        tokenizer = pickle.load(handle)
    model = load_model(config['model_path'])
    return tokenizer, model

def predict(sentence: str, tokenizer, model, maxlen: int):
    
    # Load dataset
    input_df = pd.DataFrame({'text': [sentence]}, index=[0])

    # Preprocess text
    preprocessor = PreprocessText()
    instance = preprocessor(input_df)

    # tokenizing and padding
    instance = tokenizer.texts_to_sequences(instance['preprocessed'])
    instance = pad_sequences(instance, padding='post', maxlen=maxlen)
    pred = model.predict(np.array(instance))

    # Find class with max probability
    max_prob_index = np.argmax(pred[0])

    return max_prob_index


if __name__ == "__main__":

    config = Config()

    tokenizer, model = load_data(config)
    query = input('Enter a sentence: ')
    pprint.pprint({'query': query, 'pred': predict(query, tokenizer, model, config['max_length'])})