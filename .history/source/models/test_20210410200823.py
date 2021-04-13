
import pandas as pd
from sklearn.preprocessing import LabelEncoder

df = pd.read_csv('datasets/petfinder-mini/petfinder-mini.csv')

col = ['Type', 'Breed1', 'Gender', 'Color1', 'Color2', 'MaturitySize',
       'FurLength', 'Vaccinated', 'Sterilized', 'Health',
       ]

df.drop(['Description'],axis=1,inplace=True)
df[col] = df[col].astype(str).apply(LabelEncoder().fit_transform)
#df.to_csv('datasets/petfinder-mini/petfinder-mini.csv')

df.to_parquet('datasets/petfinder_mini.parquet',index=False)
'''
import tensorflow as tf
from petastorm.tf_utils import make_petastorm_dataset
from petastorm import make_reader

with make_reader('file:///home/tushargoel/Desktop/dsa2/source/models/datasets/petfinder_mini.parquet') as reader:
    dataset = make_petastorm_dataset(reader)
    '''