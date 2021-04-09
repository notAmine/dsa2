'''
import pandas as pd
from sklearn.preprocessing import LabelEncoder

df = pd.read_csv('datasets/petfinder-mini/petfinder-mini.csv')

col = ['Type', 'Breed1', 'Gender', 'Color1', 'Color2', 'MaturitySize','FurLength', 'Vaccinated', 'Sterilized', 'Health']

df.drop(['Description'],axis=1,inplace=True)
df[col] = df[col].astype(str).apply(LabelEncoder().fit_transform)
#df.to_csv('datasets/petfinder-mini/petfinder-mini.csv')
df = df.astype('int32')
df.to_parquet('datasets/petfinder_mini.parquet',index=False)
'''
from petastorm import make_batch_reader
from petastorm.tf_utils import make_petastorm_dataset
import tensorflow as tf
from tensorflow.data.experimental import unbatch
from tensorflow.io import decode_raw
from tensorflow.keras.applications.xception import preprocess_input
from tensorflow.keras.layers import Dropout
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from pyspark.context import SparkContext
from pyspark.sql.session import SparkSession
#sc = SparkContext('local')
# spark = SparkSession(sc)

cols = 'Type,Age,Breed1,Gender,Color1,Color2,MaturitySize,FurLength,Vaccinated,Sterilized,Health,Fee,PhotoAmt,AdoptionSpeed'
cols = cols.split(',')


path = "/source/models/datasets/petfinder_mini.parquet"


def pack_features_vector(features, labels):
    """Pack the features into a single array."""
    features = tf.stack(list(features.values()), axis=1)
    return features, labels

from petastorm.tf_utils import make_petastorm_dataset
batch_size = 32
with make_batch_reader('file:///' + path ) as reader:
       dataset = make_petastorm_dataset(reader)
       dataset = dataset.make_one_shot_iterator()
       
from tensorflow.keras import layers
model = tf.keras.Sequential([
       layers.Flatten(),
       layers.Dense(256, activation='elu'),
       layers.Dense(128, activation='elu'),
       layers.Dense(32, activation='elu'),
       layers.Dense(1,activation='sigmoid') 
       ])
model.compile(optimizer='adam',
              loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
              metrics=['accuracy'])    
model.fit([dataset],
       steps_per_epoch=1,
       epochs=1,
       verbose=1
       )
print('Hurray Successfully Initiated')