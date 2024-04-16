import pandas as pd
import numpy as np
from pickle import dump

class Preprocess:
    def __init__(self, dataset):
        self.df = dataset
        self.all_columns = list(self.df.columns)
        self.unique_columns = list(self.df.select_dtypes(include=['object']).columns)
        self.unique_features = list(self.df[col].unique() for col in self.unique_columns)

    def preprocessor(self):
        self.df.replace("?", np.nan, inplace=True)
        self.df.replace(np.nan, 0, inplace=True)
        unique_id = 0
        for col in self.unique_columns:
            self.df[col].replace(self.unique_features[unique_id], list(range(0, len(self.unique_features[unique_id]))), inplace=True)
            unique_id +=1
        return self.df
    
    def save_features(self, path):
        features = [self.all_columns, self.unique_columns, self.unique_features]
        features = pd.DataFrame(features)
        dump(features, open(path + '/features.pkl', 'wb'))


