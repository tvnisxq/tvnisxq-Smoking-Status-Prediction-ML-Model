# tests/test_data_preprocessing.py

import pytest
import pandas as pd
from src.components.data_preprocessing import preprocess_data, remove_low_variance_features, select_features_by_mutual_info

@pytest.fixture
def sample_data():
    train = pd.DataFrame({
        'age': [30, 40, 35, None],
        'bmi': [22.0, 30.0, 25.5, 27.0],
        'gender': ['M', 'F', 'F', 'M'],
        'smoking': [1, 0, 1, 0]
    })
    test = pd.DataFrame({
        'age': [50, None],
        'bmi': [26.0, 24.0],
        'gender': ['F', 'M'],
        'smoking': [1, 0]
    })
    return train, test

def test_preprocess_data(sample_data):
    train_df, test_df = preprocess_data(*sample_data)
    assert not train_df.isnull().any().any()
    assert 'smoking' in train_df.columns
    assert train_df.shape[1] == test_df.shape[1]

def test_variance_removal(sample_data):
    train_df, test_df = preprocess_data(*sample_data)
    train_clean, test_clean = remove_low_variance_features(train_df, test_df)
    assert 'smoking' in train_clean.columns
    assert train_clean.shape[1] <= train_df.shape[1]

def test_mutual_info_selection(sample_data):
    train_df, test_df = preprocess_data(*sample_data)
    train_selected, test_selected = select_features_by_mutual_info(train_df, test_df, target_column='smoking', num_features=2)
    assert 'smoking' in train_selected.columns
    assert len(train_selected.columns) == 3  # 2 features + target
