import pandas as pd
from sklearn.model_selection import train_test_split


def load_data(file_path):
    column_names = ['user_id', 'item_id', 'rating', 'timestamp']
    data = pd.read_table(file_path, sep='::', names=column_names)
    print(data.info())
    return data


def preprocess_data(data):
    data = data.drop('timestamp', axis=1)
    return data


def split_data(data, test_size=0.2, random_state=42):
    train_data, test_data = train_test_split(data, test_size=test_size, random_state=random_state)
    return train_data, test_data


if __name__ == "__main__":
    file_path = r"基于横向联邦学习推荐系统/data/ratings.dat"
    data = load_data(file_path)
    data = preprocess_data(data)
    train_data, test_data = split_data(data)
    print("Train Data:\n", train_data.head())
    print("\nTest Data:\n", test_data.head())
