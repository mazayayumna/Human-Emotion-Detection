import pandas as pd
import numpy as np
import yaml
import json

from config import Config
from preprocessing import PreprocessText
from dataset import CreateDataset
from embedding import EmbeddingModel
from model import CNNModel

    
def load_dataset(dataset_path: str)-> pd.DataFrame:
    emosi = pd.read_csv(dataset_path)
    return emosi

def main(config: dict):
    # Load dataset
    emosi = load_dataset(config['dataset_path'])
    print(f'dataset columns: {emosi.columns}')
    print(f'Dataset shape: {emosi.shape}')

    # Preprocess text
    processor = PreprocessText()
    preprocessed_df = processor(emosi)

    # Create dataset
    dataset = CreateDataset(
        max_features=config['max_features'], 
        max_len=config['max_length'], 
        test_size=config['test_size'], 
        random_state=config['random_state'],
        tokenizer_path=config['tokenizer_path']
        )
    tokenizer, X_train, y_train = dataset(preprocessed_df)
    print(f'X_train shape: {X_train.shape}')
    print(f'x_train head: {X_train[0]}')
    
    # Embedding
    embedding = EmbeddingModel(max_len=config['max_length'], embedding_path=config['embedding_path'], tokenizer=tokenizer)
    embedding_matrix = embedding()
    print(f'Embedding matrix shape: {embedding_matrix.shape}')

    # CNN Model
    num_classes = len(np.unique(y_train))
    print(f'Number of classes: {num_classes}')
    cnn_model = CNNModel(
        max_len=config['max_length'], 
        embedding_matrix=embedding_matrix, 
        loss=config['loss'], 
        optimizer=config['optimizer'], 
        activation=config['activation'], 
        num_classes=num_classes
        )
    model = cnn_model()
    print(model.summary())

    # Train model
    print('begin training')
    history = model.fit(
        X_train, 
        y_train, 
        batch_size=config['batch_size'], 
        epochs=config['epochs'],
        verbose=config['verbose'],
        validation_split=config['validation_split']
        )

    # Save training history
    with open('history.json', 'w') as file:
        json.dump(history.history, file)

    #save model
    model.save(config['model_path'])

    # Evaluate model: to evaluate model, run evaluate.py.

if __name__ == '__main__':
    config = Config()
    main(config)
