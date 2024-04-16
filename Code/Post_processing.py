import numpy as np
import pandas as pd
from pickle import load

class Post_processing:
    def __init__(self, synthetic_data, path):
        self.real_feature = load(open(path + '/features.pkl', 'rb'))
        self.all_columns = list(self.real_feature.iloc[0,:])
        self.unique_columns = list(self.real_feature.iloc[1,:])
        self.unique_columns = list(filter(lambda item: item is not None, self.unique_columns))
        self.unique_features = list(self.real_feature.iloc[2,:])
        self.unique_features = list(filter(lambda item: item is not None, self.unique_features))
        self.scaler = load(open(path + '/scalling.pkl', 'rb'))
        self.synthetic_data = synthetic_data

    
    def process(self, generated_file_name, path):
        self.synthetic_data = self.scaler.inverse_transform(self.synthetic_data)
        self.synthetic_data = pd.DataFrame(self.synthetic_data, columns=self.all_columns)
        self.synthetic_data = self.synthetic_data.round(decimals=0)
        unique_id = 0
        for col in self.unique_columns:
            self.synthetic_data[col].replace(range(0, len(self.unique_features[unique_id])), self.unique_features[unique_id], inplace=True)
            unique_id +=1
        new_data = self.synthetic_data
        new_data.to_csv(path + '/' + generated_file_name + '.csv', index=False)
        return new_data