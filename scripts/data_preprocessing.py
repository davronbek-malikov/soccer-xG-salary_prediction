# scripts/data_preprocessing.py

import pandas as pd
from sklearn.preprocessing import LabelEncoder, MinMaxScaler

class SoccerDataPreprocessor:
    def __init__(self, df):
        """
        Initialize the preprocessor with a dataframe.
        """
        self.df = df.copy()
    
    def fill_missing_values(self):
        """
        Fill missing values: mode for categorical, mean for numeric.
        """
        for col in self.df.columns:
            if self.df[col].isnull().any():
                if self.df[col].dtype == 'object':
                    self.df[col].fillna(self.df[col].mode()[0], inplace=True)
                else:
                    self.df[col].fillna(self.df[col].mean(), inplace=True)
        return self
    
    def encode_categorical(self):
        """
        Encode categorical columns: use get_dummies for <5 unique values, LabelEncoder otherwise.
        """
        encoder = LabelEncoder()
        for col in self.df.columns:
            if self.df[col].dtype == 'object':
                if self.df[col].nunique() < 5:
                    dummies = pd.get_dummies(self.df[col], prefix=col, dtype=int)
                    self.df = pd.concat([self.df.drop(columns=[col]), dummies], axis=1)
                else:
                    self.df[col] = encoder.fit_transform(self.df[col])
        return self
    
    def scale_numeric(self):
        """
        Scale numeric columns (int32, int64) using MinMaxScaler, except 'salary'.
        """
        scaler = MinMaxScaler()
        num_col = self.df.select_dtypes(include=['int32','int64','float64']).columns.drop('salary', errors='ignore')
        self.df[num_col] = scaler.fit_transform(self.df[num_col])
        return self
    
    def get_data(self):
        """
        Return the preprocessed dataframe.
        """
        return self.df
