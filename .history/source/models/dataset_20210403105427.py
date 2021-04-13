import tensorflow as tf
import pandas as pd
import pprint
dst = {}
class dictEval(object):
    
    def eval_dict(self,src):
        
        
        for key, value in src.items():
            if isinstance(value, dict):
                node = dst.setdefault(key, {})
                self.eval_dict(value)
            else:
                if "@lazy" not in key :
                    dst[key] = value
                else :
                    if 'pandas.read_csv' in key :
                        key2 = key.split(":")[-1]
                        dst[key2] = pd.read_csv(value)
        return dst

test = dictEval()
dataset_url = 'http://storage.googleapis.com/download.tensorflow.org/data/petfinder-mini.zip'
csv_file    = 'datasets/petfinder-mini/petfinder-mini.csv'
tf.keras.utils.get_file('petfinder_mini.zip', dataset_url,extract=True, cache_dir='.')
src = {
    '@lazy':{
        'pandas.read_csv:petfinder':csv_file
    }
}

pprint.pprint(test.eval_dict(src))