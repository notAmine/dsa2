# pylint: disable=C0321,C0103,C0301,E1305,E1121,C0302,C0330,C0111,W0613,W0611,R1705
# -*- coding: utf-8 -*-
"""


https://github.com/arita37/pytorch_tabular
from pytorch_tabular import TabularModel
from pytorch_tabular.models import CategoryEmbeddingModelConfig
from pytorch_tabular.config import DataConfig, OptimizerConfig, TrainerConfig, ExperimentConfig

data_config = DataConfig(
    target=['target'], #target should always be a list. Multi-targets are only supported for regression. Multi-Task Classification is not implemented
    continuous_cols=num_col_names,
    categorical_cols=cat_col_names,
)
trainer_config = TrainerConfig(
    auto_lr_find=True, # Runs the LRFinder to automatically derive a learning rate
    batch_size=1024,
    max_epochs=100,
    gpus=1, #index of the GPU to use. 0, means CPU
)
optimizer_config = OptimizerConfig()

model_config = CategoryEmbeddingModelConfig(
    task="classification",
    layers="1024-512-512",  # Number of nodes in each layer
    activation="LeakyReLU", # Activation between each layers
    learning_rate = 1e-3
)

tabular_model = TabularModel(
    data_config=data_config,
    model_config=model_config,
    optimizer_config=optimizer_config,
    trainer_config=trainer_config,
)
tabular_model.fit(train=train, validation=val)
result = tabular_model.evaluate(test)
pred_df = tabular_model.predict(test)
tabular_model.save_model("examples/basic")
loaded_model = TabularModel.load_from_checkpoint("examples/basic")




"""
import os, numpy as np, pandas as pd, sklearn
from functools import partial
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

from torch import nn
import torch


try
    from pytorch_tabular import TabularModel
    from pytorch_tabular.models import CategoryEmbeddingModelConfig
    from pytorch_tabular.config import DataConfig, OptimizerConfig, TrainerConfig, ExperimentConfig
except :
    os.system("pip install pytorch_tabular[all]")


####################################################################################################
VERBOSE = False



# MODEL_URI = get_model_uri(__file__)


def log(*s):
    print(*s, flush=True)


####################################################################################################
global model, session


def init(*kw, **kwargs):
    global model, session
    model = Model(*kw, **kwargs)
    session = None



class customModel(PyroModule):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.linear = PyroModule[nn.Linear](in_features, out_features)
        self.linear.weight = PyroSample(dist.Normal(0., 1.).expand([out_features, in_features]).to_event(2))
        self.linear.bias = PyroSample(dist.Normal(0., 10.).expand([out_features]).to_event(1))

    def forward(self, x, y=None):
        sigma = pyro.sample("sigma", dist.Uniform(0., 10.))
        mean = self.linear(x).squeeze(-1)
        with pyro.plate("data", x.shape[0]):
            obs = pyro.sample("obs", dist.Normal(mean, sigma), obs=y)
        return mean



class Model(object):
    def __init__(self, model_pars=None, data_pars=None, compute_pars=None):
        self.model_pars, self.compute_pars, self.data_pars = model_pars, compute_pars, data_pars

        if model_pars is None:
            self.model = None

        else:
            ###############################################################
            """
            data_config = DataConfig(
            target=['target'], #target should always be a list. Multi-targets are only supported for regression. Multi-Task Classification is not implemented
            continuous_cols=num_col_names,
            categorical_cols=cat_col_names,
        )
        trainer_config = TrainerConfig(
            auto_lr_find=True, # Runs the LRFinder to automatically derive a learning rate
            batch_size=1024,
            max_epochs=100,
            gpus=1, #index of the GPU to use. 0, means CPU
        )
        optimizer_config = OptimizerConfig()
        
        model_config = CategoryEmbeddingModelConfig(
            task="classification",
            layers="1024-512-512",  # Number of nodes in each layer
            activation="LeakyReLU", # Activation between each layers
            learning_rate = 1e-3
        )
        
        tabular_model = TabularModel(
            data_config=data_config,
            model_config=model_config,
            optimizer_config=optimizer_config,
            trainer_config=trainer_config,
        )
        tabular_model.fit(train=train, validation=val)
        result = tabular_model.evaluate(test)
        pred_df = tabular_model.predict(test)
        tabular_model.save_model("examples/basic")
        loaded_model = TabularModel.load_from_checkpoint("examples/basic")



            """
            dm          = data_pars['cols_model_group_custom']
            data_config = DataConfig(
              target           = dm['coly'], #target should always be a list. Multi-targets are only supported for regression. Multi-Task Classification is not implemented
              continuous_cols  = dm['colnum'],
              categorical_cols = dm['colcat'],
            )


            model_config   = CategoryEmbeddingModelConfig( **model_pars['model_pars']    )
            trainer_config = TrainerConfig( **compute_pars['compute_pars'] )
            optimizer_config = OptimizerConfig()

            self.config_pars = { 'data_config' : data_config,
                        'model_config' : model_config,
                        'optimizer_config' : optimizer_config,
                        'trainer_config' : trainer_config,
            }

            self.model = TabularModel(**self.config_pars)
            self.guide = None
            self.pred_summary = None  ### All MC summary

            if VERBOSE: log(self.guide, self.model)


def fit(data_pars=None, compute_pars=None, out_pars=None, **kw):
    """
    """
    global model, session
    session = None  # Session type for compute
    Xtrain, ytrain, Xtest, ytest = get_dataset(data_pars, task_type="train")

    Xtrain = torch.tensor(Xtrain.values, dtype=torch.float)
    Xtest  = torch.tensor(Xtest.values, dtype=torch.float)
    ytrain = torch.tensor(ytrain.values, dtype=torch.float)
    ytest  = torch.tensor(ytest.values, dtype=torch.float)

    if VERBOSE: log(Xtrain, model.model)

    ###############################################################
    compute_pars2 = compute_pars.get('compute_pars', {})







    #############################################################
    df_loss = pd.DataFrame(losses)
    df_loss['loss'].plot()
    return df_loss


def predict(Xpred=None, data_pars={}, compute_pars=None, out_pars={}, **kw):
    global model, session

    compute_pars2 = model.compute_pars if compute_pars is None else compute_pars
    num_samples   = compute_pars2.get('num_samples', 300)

    ###### Data load
    if Xpred is None:
        Xpred = get_dataset(data_pars, task_type="predict")
    cols_Xpred = list(Xpred.columns)

    max_size = compute_pars2.get('max_size', len(Xpred))
    Xpred    = Xpred.iloc[:max_size, :]
    Xpred_   = torch.tensor(Xpred.values, dtype=torch.float)

    ###### Post processing normalization
    post_process_fun = model.model_pars.get('post_process_fun', None)
    if post_process_fun is None:
        def post_process_fun(y):
            return y


    #####################################################################


    ypred = model.model.predict









    #################################################################
    model.pred_summary = {'pred_mean': ypred_mean, 'pred_summary': pred_summary, 'pred_samples': pred_samples}
    print('stored in model.pred_summary')
    # print(  dd['y_mean'], dd['y_mean'].shape )
    # import pdb; pdb.set_trace()
    return dd['y_mean']


def reset():
    global model, session
    model, session = None, None


def save(path=None, info=None):
    """
       Custom saving
    """

    ""
    global model, session
    import cloudpickle as pickle
    os.makedirs(path, exist_ok=True)

    filename = "model.pkl"
    pickle.dump(model, open(f"{path}/{filename}", mode='wb'))  # , protocol=pickle.HIGHEST_PROTOCOL )

    filename = "info.pkl"
    pickle.dump(info, open(f"{path}/{filename}", mode='wb'))  # ,protocol=pickle.HIGHEST_PROTOCOL )


def load_model(path=""):
    global model, session
    import cloudpickle as pickle
    model0 = pickle.load(open(f"{path}/model.pkl", mode='rb'))

    model = Model()  # Empty model
    model.model = model0.model
    model.model_pars = model0.model_pars
    model.compute_pars = model0.compute_pars
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

        X, y = make_classification(n_features=10, n_redundant=0, n_informative=2,
                                   random_state=1, n_clusters_per_class=1)

        # log(X,y)
        Xtrain, Xtest, ytrain, ytest = train_test_split(X, y)
        return Xtrain, ytrain, Xtest, ytest

    if prepro_pars['type'] == 'train':
        from sklearn.model_selection import train_test_split
        df = pd.read_csv(prepro_pars['path'])
        dfX = df[prepro_pars['colX']]
        dfy = df[prepro_pars['coly']]
        Xtrain, Xtest, ytrain, ytest = train_test_split(dfX.values, dfy.values)
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





def test(config=''):
    """
       python model_torch_tabular.py test

    """
    global model, session

    X = np.random.rand(10000,20)
    y = np.random.binomial(n=1, p=0.5, size=[10000])

    X_train_full, X_test, y_train_full, y_test = train_test_split(X, y, random_state=2021, stratify=y)
    X_train, X_valid, y_train, y_valid         = train_test_split(X_train_full, y_train_full, random_state=2021, stratify=y_train_full)


    model_pars = {'model_class': 'WideAndDeep',
                  'model_pars': {'n_wide_cross': 10,
                                 'n_wide': 10},
                 }
    data_pars = {'train': {'Xtrain': X_train,
                           'ytrain': y_train,
                           'Xtest': X_test,
                           'ytest': y_test},
                 'eval': {'X': X_valid,
                          'y': y_valid},
                 'predict': {'X': X_valid},
                }
    compute_pars = { 'compute_pars' : { 'epochs': 50,
                    'callbacks': callbacks} }



    model = Model(model_pars=model_pars, data_pars=data_pars, compute_pars=compute_pars)

    print('\n\nTraining the model..')
    fit(data_pars=data_pars, compute_pars=compute_pars, out_pars=None)
    print('Training completed!\n\n')

    print('Predict data..')
    ypred, ypred_proba = predict(Xpred=None, data_pars=data_pars, compute_pars=compute_pars)
    print(f'Top 5 y_pred: {np.squeeze(ypred)[:5]}')
    print('Data successfully predicted!\n\n')

    print('Evaluating the model..')
    print(eval(data_pars=data_pars, compute_pars=compute_pars))
    print('Evaluating completed!\n\n')

    print('Saving model..')
    save(path='model_dir/')
    print('Model successfully saved!\n\n')

    print('Load model..')
    model, session = load_model(path="model_dir/")
    print('Model successfully loaded!\n\n')

    print('Model architecture:')
    print(model)







if __name__ == "__main__":
    import fire
    fire.Fire(test)
