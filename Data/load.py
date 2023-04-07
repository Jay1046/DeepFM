import pandas as pd


def load_data(path):
    
    data = pd.read_csv(path).drop(['Unnamed: 0', 'img_link'], axis=1)
    
    return data