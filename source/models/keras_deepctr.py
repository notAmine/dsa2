""""

Most difficult part is pre-processing.

# DeepCTR
https://github.com/shenweichen/DeepCTR
https://deepctr-doc.readthedocs.io/en/latest/Examples.html#classification-criteo


DeepCTR is a **Easy-to-use**,**Modular** and **Extendible** package of deep-learning based CTR models
along with lots of core components layers which can be used to easily build custom models.It is compatible with **tensorflow 1.4+ and 2.0+**.You can use any complex model with `model.fit()`and `model.predict()` .


## Models List
|                 Model                  | Paper                                                                                                                                                           |
| :------------------------------------: | :-------------------------------------------------------------------------------------------------------------------------------------------------------------- |
|  Convolutional Click Prediction Model  | [CIKM 2015][A Convolutional Click Prediction Model](http://ir.ia.ac.cn/bitstream/173211/12337/1/A%20Convolutional%20Click%20Prediction%20Model.pdf)             |
| Factorization-supported Neural Network | [ECIR 2016][Deep Learning over Multi-field Categorical Data: A Case Study on User Response Prediction](https://arxiv.org/pdf/1601.02376.pdf)                    |
|      Product-based Neural Network      | [ICDM 2016][Product-based neural networks for user response prediction](https://arxiv.org/pdf/1611.00144.pdf)                                                   |
|              Wide & Deep               | [DLRS 2016][Wide & Deep Learning for Recommender Systems](https://arxiv.org/pdf/1606.07792.pdf)                                                                 |
|                 DeepFM                 | [IJCAI 2017][DeepFM: A Factorization-Machine based Neural Network for CTR Prediction](http://www.ijcai.org/proceedings/2017/0239.pdf)                           |
|        Piece-wise Linear Model         | [arxiv 2017][Learning Piece-wise Linear Models from Large Scale Data for Ad Click Prediction](https://arxiv.org/abs/1704.05194)                                 |
|          Deep & Cross Network          | [ADKDD 2017][Deep & Cross Network for Ad Click Predictions](https://arxiv.org/abs/1708.05123)                                                                   |
|   Attentional Factorization Machine    | [IJCAI 2017][Attentional Factorization Machines: Learning the Weight of Feature Interactions via Attention Networks](http://www.ijcai.org/proceedings/2017/435) |
|      Neural Factorization Machine      | [SIGIR 2017][Neural Factorization Machines for Sparse Predictive Analytics](https://arxiv.org/pdf/1708.05027.pdf)                                               |
|                xDeepFM                 | [KDD 2018][xDeepFM: Combining Explicit and Implicit Feature Interactions for Recommender Systems](https://arxiv.org/pdf/1803.05170.pdf)                         |
|                AutoInt                 | [arxiv 2018][AutoInt: Automatic Feature Interaction Learning via Self-Attentive Neural Networks](https://arxiv.org/abs/1810.11921)                              |
|         Deep Interest Network          | [KDD 2018][Deep Interest Network for Click-Through Rate Prediction](https://arxiv.org/pdf/1706.06978.pdf)                                                       |
|    Deep Interest Evolution Network     | [AAAI 2019][Deep Interest Evolution Network for Click-Through Rate Prediction](https://arxiv.org/pdf/1809.03672.pdf)                                            |
|                  NFFM                  | [arxiv 2019][Operation-aware Neural Networks for User Response Prediction](https://arxiv.org/pdf/1904.12579.pdf)                                                |
|                 FGCNN                  | [WWW 2019][Feature Generation by Convolutional Neural Network for Click-Through Rate Prediction ](https://arxiv.org/pdf/1904.04447)                             |
|     Deep Session Interest Network      | [IJCAI 2019][Deep Session Interest Network for Click-Through Rate Prediction ](https://arxiv.org/abs/1905.06482)                                                |
|                FiBiNET                 | [RecSys 2019][FiBiNET: Combining Feature Importance and Bilinear feature Interaction for Click-Through Rate Prediction](https://arxiv.org/pdf/1905.09433.pdf)   |


Names"

model_list = ["AFM",
"AUTOINT",
"CCPM",
"DCN",
"DeepFM",
"DIEN",
"DIN",
"DSIN",
"FGCNN",
"FIBINET",
"FLEN",
"FNN",
"MLR",
"NFM",
"ONN",
"PNN",
"WDL",
"XDEEPFM", ]


"""
from jsoncomment import JsonComment ; json = JsonComment()
import os
from pathlib import Path
import importlib


import numpy as np
import pandas as pd
import keras
from keras.preprocessing.sequence import pad_sequences
from keras.callbacks import EarlyStopping, ModelCheckpoint
import sklearn
from sklearn.metrics import log_loss, mean_squared_error, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, MinMaxScaler

import deepctr
from deepctr.feature_column import DenseFeat, SparseFeat, VarLenSparseFeat, get_feature_names
from deepctr.models import DeepFM



# from preprocess import _preprocess_criteo, _preprocess_movielens

# Note: keep that to disable eager mode with tf 2.x
import tensorflow as tf
if tf.__version__ >= '2.0.0':
    tf.compat.v1.disable_eager_execution()


####################################################################################################
# Helper functions
#from mlmodels.util import os_package_root_path, log, path_norm
#from mlmodels.util import save_keras, load_keras
#from mlmodels.preprocess.tabular_keras  import get_test_data, get_xy_fd_dien, get_xy_fd_din, get_xy_fd_dsin


def log(*s):
    print(s, flush=True)


####################################################################################################
DATA_PARAMS = {
    "AFM"     : {"sparse_feature_num": 3, "dense_feature_num": 0},
    "AutoInt" : {"sparse_feature_num": 1, "dense_feature_num": 1},
    "CCPM"    : {"sparse_feature_num": 3, "dense_feature_num":0},
    "DCN"     : {"sparse_feature_num": 3, "dense_feature_num": 3},
    "DeepFM"  : {"sparse_feature_num": 1, "dense_feature_num": 1},
    "DIEN"    : {},
    "DIN"     : {},
    "DSIN"    : {},
    "FGCNN"   : {"embedding_size": 8, "sparse_feature_num": 1, "dense_feature_num": 1},
    "FiBiNET" : {"sparse_feature_num": 2, "dense_feature_num": 2},
    "FLEN"    : {"embedding_size": 2, "sparse_feature_num": 6, "dense_feature_num": 6, "use_group": True},
    "FNN"     : {"sparse_feature_num": 1, "dense_feature_num": 1},
    "MLR"     : {"sparse_feature_num": 0, "dense_feature_num": 2, "prefix": "region"},
    "NFM"     : {"sparse_feature_num": 1, "dense_feature_num": 1},
    "ONN"     : {"sparse_feature_num": 2, "dense_feature_num": 2, "sequence_feature":('sum', 'mean', 'max',), "hash_flag":True},
    "PNN"     : {"sparse_feature_num": 1, "dense_feature_num": 1},
    "WDL"     : {"sparse_feature_num": 2, "dense_feature_num": 0},
    "xDeepFM" : {"sparse_feature_num": 1, "dense_feature_num": 1}
}

MODEL_PARAMS = {
    "AFM"     : {"use_attention": True, "afm_dropout": 0.5},
    "AutoInt" : {"att_layer_num": 1, "dnn_hidden_units": (), "dnn_dropout": 0.5},
    "CCPM"    : {"conv_kernel_width": (3, 2), "conv_filters": (2, 1), "dnn_hidden_units": [32,], "dnn_dropout": 0.5},
    "DCN"     : {"cross_num": 0, "dnn_hidden_units": (8,), "dnn_dropout": 0.5},
    "DeepFM"  : {"dnn_hidden_units": (2,), "dnn_dropout": 0.5},
    "DIEN"    : {"dnn_hidden_units": [4, 4, 4], "dnn_dropout": 0.5, "gru_type": "GRU"},
    "DIN"     : {"dnn_hidden_units":[4, 4, 4], "dnn_dropout":0.5},
    "DSIN"    : {"sess_max_count":2, "dnn_hidden_units":[4, 4, 4], "dnn_dropout":0.5},
    "FGCNN"   : {"conv_kernel_width":(3,2), "conv_filters":(2, 1), "new_maps":(2, 2), "pooling_width":(2, 2), "dnn_hidden_units": (32, ), "dnn_dropout":0.5},
    "FiBiNET" : {"bilinear_type": "all", "dnn_hidden_units":[4,], "dnn_dropout":0.5},
    "FLEN"    : {"dnn_hidden_units": (3,), "dnn_dropout":0.5},
    "FNN"     : {"dnn_hidden_units":[32, 32], "dnn_dropout":0.5},
    "MLR"     : {},
    "NFM"     : {"dnn_hidden_units":[32, 32], "dnn_dropout":0.5},
    "ONN"     : {"dnn_hidden_units": [32, 32], "embedding_size":4, "dnn_dropout":0.5},
    "PNN"     : {"embedding_size":4, "dnn_hidden_units":[4, 4], "dnn_dropout":0.5, "use_inner": True, "use_outter": True},
    "WDL"     : {"dnn_hidden_units":[32, 32], "dnn_dropout":0.5},
    "xDeepFM" : {"dnn_dropout": 0.5, "dnn_hidden_units": (8,), "cin_layer_size": (), "cin_split_half": True, "cin_activation": 'linear'}
}

class Model:
    def __init__(self, model_pars=None, data_pars=None, compute_pars=None, **kwargs):
        self.model_pars, self.compute_pars, self.data_pars = model_pars, compute_pars, data_pars
        self.history = None        
        if model_pars is None :
          return self

        model_name = model_pars.get("model_name", "DeepFM")
        model_list = list(MODEL_PARAMS.keys())
      
        # assert model_name in model_list, raise ValueError('Not existing model', model_name)
        modeli     = getattr(importlib.import_module("deepctr.models"), model_name)

        # 4.Define Model #################################################
        model_params = model_pars.get('model_pars', MODEL_PARAMS[model_name] )

        if model_name in ["DIEN", "DIN", "DSIN"]:

             feature_cols = None
             behavior_feature_list = [] 

             self.model = modeli(feature_cols, behavior_feature_list, **model_params )

        elif model_name == "MLR":
             feature_cols = None

             self.model = modeli(feature_cols)


        elif model_name == "PNN":
             feature_cols = None

             self.model = modeli(feature_cols, **MODEL_PARAMS[model_name])
        
        else:  # ['WDL' ] :
             linear_cols = model_pars.get('linear_feature_cols', None)
             dnn_cols    = model_pars.get('dnn_feature_cols', None)
             self.model = modeli(linear_cols, dnn_cols, **model_params )

        #if model_name in ['WDL' ] : 
        #    self.model = modeli(linear_feature_cols, dnn_feature_cols, **MODEL_PARAMS[model_name])


        #################################################################################
        self.model.compile(optimizer=model_pars['optimization'],
                           loss=model_pars['loss'],
                           metrics=compute_pars.get("metrics", ['binary_crossentropy']), )
        self.model.summary()






def fit(data_pars=None, compute_pars=None, out_pars=None, **kw):
    """
    """
    global model, session
    session = None  # Session type for compute
    Xtrain, ytrain, Xtest, ytest = get_dataset(data_pars, task_type="train")

    # if VERBOSE: log(Xtrain.shape, model.model)

    cpars = compute_pars.get("compute_pars", {})
    assert 'epochs' in cpars, 'epoch'

    hist = model.model.fit(Xtrain, ytrain, **cpars)
    model.history = hist


def eval(data_pars=None, compute_pars=None, out_pars=None, **kw):
    """
       Return metrics of the model when fitted.
    """
    global model, session
    data_pars['train'] = True
    Xval, yval = get_dataset(data_pars, task_type="eval")
    ypred = predict(Xval, data_pars, compute_pars, out_pars)

    # log(data_pars)
    mpars = compute_pars.get("metrics_pars", {'metric_name': 'mae'})

    scorer = {
        "rmse": sklearn.metrics.mean_squared_error,
        "mae": sklearn.metrics.mean_absolute_error
    }[mpars['metric_name']]

    mpars2 = mpars.get("metrics_pars", {})  ##Specific to score
    score_val = scorer(yval, ypred[0], **mpars2)

    ddict = [{"metric_val": score_val, 'metric_name': mpars['metric_name']}]

    return ddict


def predict(Xpred=None, data_pars={}, compute_pars={}, out_pars={}, **kw):
    global model, session

    Xpred = get_dataset(data_pars, task_type="predict")
    ypred = model.model.predict(Xpred)

    ypred_proba = None  ### No proba
    if compute_pars.get("probability", False):
         ypred_proba = model.model.predict_proba(Xpred)
    return ypred, ypred_proba


def reset():
    global model, session
    model, session = None, None


def save(path=None):
    global model, session
    os.makedirs(path, exist_ok=True)

    filename = "model.h5"
    filepath = path + filename
    keras.models.save_model(model.model, filepath)
    # model.model.save(filepath)


def load_model(path=""):
    global model, session

    filepath = path + 'model.h5'
    model = keras.models.load_model(filepath, deepctr.layers.custom_objects)
    session = None
    return model, session


def load_info(path=""):
    import cloudpickle as pickle, glob
    dd = {}
    for fp in glob.glob(f"{path}/*.pkl"):
        if not "model.pkl" in fp:
            obj = pickle.load(open(fp, mode='rb'))
            key = fp.split("/")[-1]
            dd[key] = obj
    return dd


def preprocess(prepro_pars):
    if prepro_pars['type'] == 'test':
        from sklearn.datasets import make_classification
        from sklearn.model_selection import train_test_split

        X, y = make_classification(n=10, n_redundant=0, n_informative=2,
                                   random_state=1, n_clusters_per_class=1)

        # log(X,y)
        Xtrain, Xtest, ytrain, ytest = train_test_split(X, y)
        return Xtrain, ytrain, Xtest, ytest

    if prepro_pars['type'] == 'train':
        from sklearn.model_selection import train_test_split
        df = pd.read_csv(prepro_pars['path'])
        dfX = df[prepro_pars['colX']]
        dfy = df[prepro_pars['coly']]
        Xtrain, Xtest, ytrain, ytest = train_test_split(dfX.values, dfy.values,
                                                        stratify=dfy.values,test_size=0.1)
        return Xtrain, ytrain, Xtest, ytest

    else:
        df = pd.read_csv(prepro_pars['path'])
        dfX = df[prepro_pars['colX']]

        Xtest, ytest = dfX, None
        return None, None, Xtest, ytest


####################################################################################################
############ Do not change #########################################################################
def get_dataset(data_pars=None, task_type="train", **kw):
    """
      "ram"  :
      "file" :
    """
    # log(data_pars)
    data_type = data_pars.get('type', 'ram')
    if data_type == "ram":
        if task_type == "predict":
            d = data_pars[task_type]
            return d["X"]

        if task_type == "eval":
            d = data_pars[task_type]
            return d["X"], d["y"]

        if task_type == "train":
            d = data_pars[task_type]
            return d["Xtrain"], d["ytrain"], d["Xtest"], d["ytest"]

    elif data_type == "file":
        raise Exception(f' {data_type} data_type Not implemented ')

    raise Exception(f' Requires  Xtrain", "Xtest", "ytrain", "ytest" ')


def get_params_sklearn(deep=False):
    return model.model.get_params(deep=deep)


def get_params(param_pars={}, **kw):
    import json
    # from jsoncomment import JsonComment ; json = JsonComment()
    pp = param_pars
    choice = pp['choice']
    config_mode = pp['config_mode']
    data_path = pp['data_path']

    if choice == "json":
        cf = json.load(open(data_path, mode='r'))
        cf = cf[config_mode]
        return cf['model_pars'], cf['data_pars'], cf['compute_pars'], cf['out_pars']

    else:
        raise Exception(f"Not support choice {choice} yet")

########################################################################################################################




########################################################################################################################
def test(config=''):
    """

     colcat  ---. Sparse Features

     colnum --> Dense features

    """
    global model, session

    X = np.random.rand(100,30)
    y = np.random.binomial(n=1, p=0.5, size=[100])

    ## PREPROCESSING STEPS
    # change into dataframe
    cols                   = [str(i) for i in range(X.shape[1])]  # define column pd dataframe, need to be string type
    data                   = pd.DataFrame(X, columns=cols)  # need to convert into df, following the step from documentation
    data['y']              = y

    # define which feature columns sparse or dense type
    # since our data categorize as Dense Features, we define the sparse features as empty list
    cols_sparse   = []
    cols_dense    = [str(i) for i in range(X.shape[1])]

    # convert feature type into SparseFeat or DenseFeat type, adjusting from DeepCTR library
    sparse_feat_l          = [SparseFeat(feat, vocabulary_size=data[feat].nunique(), embedding_dim=4) for i,feat
                              in enumerate(cols_sparse) ]

    dense_feat_l           = [DenseFeat(feat, dimension=1) for feat in cols_dense]

    feature_cols        = sparse_feat_l +  dense_feat_l


    linear_feature_cols    = feature_cols  # containing all the features used by linear part of the model
    dnn_feature_cols       = feature_cols  # containing all the features used by deep part of the model
    feature_names          = get_feature_names(linear_feature_cols + dnn_feature_cols)

    train_full, test       = train_test_split(data, random_state=2021, stratify=data['y'])
    train, val             = train_test_split(train_full, random_state=2021, stratify=train_full['y'])

    train_model_input      = {name:train[name] for name in feature_names}
    val_model_input        = {name:val[name]   for name in feature_names}
    test_model_input       = {name:test[name]  for name in feature_names}
    target                 = 'y'
    #### END OF PREPROCESSING STEPS

    #### initalize model_pars
    opt = keras.optimizers.Adam()
    loss = 'binary_crossentropy'


    # initialize compute_pars
    early_stopping = EarlyStopping(monitor='loss', patience=3)
    # Note: ModelCheckpoint error when used
    # model_ckpt = ModelCheckpoint(filepath='', save_best_only=True, monitor='loss')
    callbacks = [early_stopping]


    for model_name in  ['WDL'] :
        ######## Model Dict  ########################################################
        model_pars = {'model_name': model_name,
                      'linear_feature_cols'    : linear_feature_cols,
                      'dnn_feature_cols'       : dnn_feature_cols,

                      'optimization'           : opt,
                      'loss'                   : loss}

        data_pars = {'train': {'Xtrain': train_model_input,
                               'ytrain': train[target].values,
                               'Xtest':  test_model_input,
                               'ytest':  test[target].values},

                     'eval': {'X': val_model_input,
                              'y': val[target].values},
                     'predict': {'X': val_model_input},
                    }

        # compute_pars = {}
        compute_pars = {'compute_pars': {'epochs': 50,
                        'callbacks': callbacks} }

        test_helper(model_pars, data_pars, compute_pars)




def test_helper(model_pars, data_pars, compute_pars):

    model = Model(model_pars=model_pars, data_pars=data_pars, compute_pars=compute_pars)

    log('\n\nTraining the model..')
    fit(data_pars=data_pars, compute_pars=compute_pars, out_pars=None)
    log('Training completed!\n\n')

    log('Predict data..')
    ypred, ypred_proba = predict(Xpred=None, data_pars=data_pars, compute_pars=compute_pars)
    log(f'Top 5 y_pred: {np.squeeze(ypred)[:5]}')
    log('Data successfully predicted!\n\n')

    log('Evaluating the model..')
    log(eval(data_pars=data_pars, compute_pars=compute_pars))
    log('Evaluating completed!\n\n')
    #
    log('Saving model..')
    save(path='model_dir/')
    log('Model successfully saved!\n\n')

    log('Load model..')
    model, session = load_model(path="model_dir/")
    log('Model successfully loaded!\n\n')

    log('Model architecture:')
    log(model.summary())



if __name__ == '__main__':
    import fire
    fire.Fire()








########################################################################################################################
########################################################################################################################



















"""


    VERBOSE = True
    for model_name in MODEL_PARAMS.keys():
        if model_name == "FGCNN": # TODO: check save io
            continue
        test(pars_choice=5, **{"model_name": model_name})

    # test(pars_choice=1)
    # test(pars_choice=2)
    # test(pars_choice=3)
    # test(pars_choice=4)



"""



