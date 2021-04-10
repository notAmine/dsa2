import tensorflow as tf
<<<<<<< HEAD
import pandas as pd
import pprint
import zipfile
dst = {}
class dictEval(object):
    
    def eval_dict(self,src):
        
        
=======
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,DenseFeatures
import pandas as pd
import pprint
import zipfile
import numpy as np
from sklearn.preprocessing import LabelEncoder
from glob import glob



def pack_features_vector(features, labels):
    """Pack the features into a single array."""
    features = tf.stack(list(features.values()), axis=1)
    return features, labels
class dictEval(object):
    global dst
    def eval_dict(self,src):
>>>>>>> 8bf95fea003f935efd1d572ff8b9a7fceaf23a75
        for key, value in src.items():
            if isinstance(value, dict):
                node = dst.setdefault(key, {})
                self.eval_dict(value)
            else:
                if "@lazy" not in key :
                    dst[key] = value
                else :
<<<<<<< HEAD
                    if 'pandas' in key :
                        key2 = key.split(":")[-1]
                        ext = value.split('.')[-1]
                        if ext == 'zip':
                            zf = zipfile.ZipFile(value)
                            dst[key2] = pd.read_csv(zf.open(src['filename']))
                        elif ext in ['csv','txt']:
                            dst[key2] = pd.read_csv(value)
                        elif ext == 'parquet':
                            dst[key2] = pd.read_parquet(value)
        return dst

test = dictEval()

dataset_url = 'http://storage.googleapis.com/download.tensorflow.org/data/petfinder-mini.zip'
csv_file    = 'datasets/petfinder-mini/petfinder-mini.csv'
zip_file = 'datasets/petfinder_mini.zip'
tf.keras.utils.get_file('petfinder_mini.zip', dataset_url,extract=True, cache_dir='.')

data_pars = {
    'train':{
        'filename':'petfinder-mini/petfinder-mini.csv', #If Zip file contains more than one file
        '@lazy_pandas:Xtrain':csv_file, #CSV file extraction
        '@lazy_pandas:Xtest':zip_file, #zip File Extraction
        '@lazy_tfdataset:Xtest':csv_file,
    },
    'pars': 23,
    
     "batch_size" : 5000,
    "n_train": 500, 
    "n_test": 500
}

dic = test.eval_dict(data_pars)
pprint.pprint(dic)
=======
                    key2 = key.split(':')[-1]
                    ext = value.split('.')[-1]
                    target = src.get('target','y')


                    ##################################################################################################
                    if 'tf' in key :
                        print('TF is HEre')
                        self.tf_dataset_create(ext,key2,target,value)
                        return dst



                    ##################################################################################################
                    if 'pandas' in key :
                        self.pandas_create(ext,key2,target,value)
                        return dst


                    ##################################################################################################
                    if 'pandas' in key :
                        pass

    def tf_dataset_create(self, ext,key2,target,value):
                        if ext == 'zip':
                            zf = zipfile.ZipFile(value)
                            fileNames = zf.namelist()
                            for idx,file in enumerate(fileNames):

                                if file.split('.')[-1] in ['csv','txt']:
                                    file = 'datasets/'+file
                                    try:
                                        dataset = tf.data.experimental.make_csv_dataset(file,label_name=target, batch_size=32,ignore_errors=True)
                                        dataset = dataset.map(pack_features_vector)
                                        dst[key2+'_'+str(idx)] = dataset.repeat()
                                    except:
                                        pass

                        elif ext in ['csv','txt']:
                                    dataset = tf.data.experimental.make_csv_dataset(value, label_name=target,batch_size=32,ignore_errors=True)
                                    dataset = dataset.map(pack_features_vector)
                                    dst[key2] = dataset.repeat()

                        elif ext == 'parquet':
                                filename = value.split('.')[0]+'.csv'
                                pd.read_parquet(value).to_csv(filename)
                                dataset = tf.data.experimental.make_csv_dataset(filename, label_name=target,batch_size=32,ignore_errors=True)
                                dataset = dataset.map(pack_features_vector)
                                dst[key2] = dataset.repeat()


    def pandas_create(self,ext,key2,target,value):
                        if ext == 'zip':
                            zf = zipfile.ZipFile(value)
                            fileNames = zf.namelist()
                            for idx,file in enumerate(fileNames):

                                if file.split('.')[-1] in ['csv','txt']:
                                    file = 'datasets/'+file
                                    dst[key2+'_'+str(idx)] = pd.read_csv(file)



                        elif ext in ['csv','txt']:
                                dst[key2] = pd.read_csv(value)

                        elif ext == 'parquet':
                                dst[key2] = pd.read_parquet(value)
                        return dst





if __name__ == '__main__':
    test = dictEval()
    dst = {}
    dataset_url = 'http://storage.googleapis.com/download.tensorflow.org/data/petfinder-mini.zip'
    csv_file    = 'datasets/petfinder-mini/petfinder-mini.csv'
    zip_file = 'datasets/petfinder_mini.zip'
    #tf.keras.utils.get_file('petfinder_mini.zip', dataset_url,extract=True, cache_dir='.')
    
    #Uncomment This File for preprocessing the CSV File
    '''df = pd.read_csv('datasets/petfinder-mini/petfinder-mini.csv')

    col = ['Type', 'Breed1', 'Gender', 'Color1', 'Color2', 'MaturitySize',
        'FurLength', 'Vaccinated', 'Sterilized', 'Health',
        ]

    df.drop(['Description'],axis=1,inplace=True)
    df[col] = df[col].astype(str).apply(LabelEncoder().fit_transform)
    df.to_csv('datasets/petfinder-mini/petfinder-mini.csv')'''
    
    
    # parquet_file = 'datasets/petfinder_mini.parquet'
    # txt_file = '/home/tushargoel/Desktop/sample.txt'

    parquet_path = 'datasets/petfinder_mini.parquet'
    txt_path = '/home/tushargoel/Desktop/sample.txt'

    csv_path    = 'datasets/petfinder-mini/petfinder-mini.csv'
    zip_path = 'datasets/petfinder_mini.zip'


    data_pars = {
        'train':{
            'target': 'Type',
            '@lazy_tf:Xtrain':parquet_path, #CSV file extraction #Tensorflow Dataset
            '@lazy_tf:Xtest':zip_path, #zip File Extraction
            '@lazy_pandas:Xtest':txt_path, #Pandas
        },
        'pars': 23,
        
        "batch_size" : 32,
        "n_train": 500, 
        "n_test": 500
    }


    test.eval_dict(data_pars)
   
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
    model.fit(dst['Xtrain'],
            steps_per_epoch=1,
            epochs=1,
            verbose=1
            )

>>>>>>> 8bf95fea003f935efd1d572ff8b9a7fceaf23a75
