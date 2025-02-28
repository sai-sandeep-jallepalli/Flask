import os

BASE_DIR = os.path.abspath(os.path.dirname(__file__))

CONFIG = {
    'BASE_DIR': BASE_DIR,
    'DATASET_PATH': os.path.join(BASE_DIR, '../datasets/Heart-Disease-Dataset.csv'),
    'MODEL_PATH': os.path.join(BASE_DIR, '../pickle-models/train.pkl')
}