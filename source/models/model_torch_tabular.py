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

import torch
from torch import nn


try :
    from pytorch_tabular import TabularModel
    from pytorch_tabular.models import CategoryEmbeddingModelConfig
    from pytorch_tabular.config import DataConfig, OptimizerConfig, TrainerConfig, ExperimentConfig
except :
    os.system("pip install pytorch_tabular[all]")


import torch
import numpy as np
from torch.functional import norm
# torch.manual_seed(0)
# np.random.seed(0)
# torch.set_deterministic(True)

from sklearn.datasets import fetch_covtype

# from torch.utils import data
from pytorch_tabular.config import (
    DataConfig,
    ExperimentConfig,
    ExperimentRunManager,
    ModelConfig,
    OptimizerConfig,
    TrainerConfig,
)
from pytorch_tabular.models.node.config import NodeConfig
from pytorch_tabular.models.category_embedding.config import (
    CategoryEmbeddingModelConfig,
)
from pytorch_tabular.models.category_embedding.category_embedding_model import (
    CategoryEmbeddingModel,
)
import pandas as pd
from omegaconf import OmegaConf
from pytorch_tabular.tabular_datamodule import TabularDatamodule
from pytorch_tabular.tabular_model import TabularModel
import pytorch_lightning as pl
from sklearn.preprocessing import PowerTransformer
from sklearn.model_selection import train_test_split
from pathlib import Path
import wget


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

            model_config     = CategoryEmbeddingModelConfig( **model_pars['model_pars']    )
            trainer_config   = TrainerConfig( **compute_pars['compute_pars'] )
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

    #Xtrain = torch.tensor(Xtrain.values, dtype=torch.float)
    #Xtest  = torch.tensor(Xtest.values, dtype=torch.float)
    #ytrain = torch.tensor(ytrain.values, dtype=torch.float)
    #ytest  = torch.tensor(ytest.values, dtype=torch.float)

    train = pd.concat((Xtrain,ytest)).values
    val   = pd.concat((Xtrain,ytest)).values


    if VERBOSE: log(Xtrain, model.model)

    ###############################################################
    compute_pars2 = compute_pars.get('compute_pars', {})


    ypred =  model.model.fittrain=train, validation=val)




    #############################################################
    return ypred


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

    #X = np.random.rand(10000,20)
    #y = np.random.binomial(n=1, p=0.5, size=[10000])

    BASE_DIR = Path.home().joinpath('data/input/covtype/')
    datafile = BASE_DIR.joinpath('covtype.data.gz')
    datafile.parent.mkdir(parents=True, exist_ok=True)
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/covtype/covtype.data.gz"
    if not datafile.exists():
        wget.download(url, datafile.as_posix())

    target_name = ["Covertype"]

    cat_col_names = [ "Wilderness_Area1", "Wilderness_Area2", "Wilderness_Area3", "Wilderness_Area4", "Soil_Type1", "Soil_Type2", "Soil_Type3", "Soil_Type4", "Soil_Type5", "Soil_Type6", "Soil_Type7", "Soil_Type8", "Soil_Type9", "Soil_Type10", "Soil_Type11", "Soil_Type12", "Soil_Type13", "Soil_Type14", "Soil_Type15", "Soil_Type16", "Soil_Type17", "Soil_Type18", "Soil_Type19", "Soil_Type20", "Soil_Type21", "Soil_Type22", "Soil_Type23", "Soil_Type24", "Soil_Type25", "Soil_Type26", "Soil_Type27", "Soil_Type28", "Soil_Type29", "Soil_Type30", "Soil_Type31", "Soil_Type32", "Soil_Type33", "Soil_Type34", "Soil_Type35", "Soil_Type36", "Soil_Type37", "Soil_Type38", "Soil_Type39", "Soil_Type40"
                      ]

    num_col_names = [ "Elevation", "Aspect", "Slope", "Horizontal_Distance_To_Hydrology", "Vertical_Distance_To_Hydrology", "Horizontal_Distance_To_Roadways", "Hillshade_9am", "Hillshade_Noon", "Hillshade_3pm", "Horizontal_Distance_To_Fire_Points"
    ]

    feature_columns = (
        num_col_names + cat_col_names + target_name)

    df = pd.read_csv(datafile, header=None, names=feature_columns)
    # cat_col_names = []

    # num_col_names = [
    #     "Elevation", "Aspect"
    # ]
    # feature_columns = (
    #     num_col_names + cat_col_names + target_name)
    # df = df.loc[:,feature_columns]
    df.head()
    train, test = train_test_split(df, random_state=42)
    train, val = train_test_split(train, random_state=42)
    num_classes = len(set(train[target_name].values.ravel()))


    data_config = DataConfig(
        target=target_name,
        continuous_cols=num_col_names,
        categorical_cols=cat_col_names,
        continuous_feature_transform=None,#"quantile_normal",
        normalize_continuous_features=False
    )
    model_config = CategoryEmbeddingModelConfig(task="classification",
                                                metrics=["f1","accuracy"],
                                                metrics_params=[{"num_classes":num_classes},{}])
    # model_config = NodeConfig(
    #     task="classification",
    #     depth=4,
    #     num_trees=1024,
    #     input_dropout=0.0,
    #     metrics=["f1", "accuracy"],
    #     metrics_params=[{"num_classes": num_classes, "average": "macro"}, {}],
    # )
    trainer_config = TrainerConfig(gpus=1, fast_dev_run=True)
    experiment_config = ExperimentConfig(project_name="PyTorch Tabular Example",
                                         run_name="node_forest_cov",
                                         exp_watch="gradients",
                                         log_target="wandb",
                                         log_logits=True)
    optimizer_config = OptimizerConfig()

    # tabular_model = TabularModel(
    #     data_config="examples/data_config.yml",
    #     model_config="examples/model_config.yml",
    #     optimizer_config="examples/optimizer_config.yml",
    #     trainer_config="examples/trainer_config.yml",
    #     # experiment_config=experiment_config,
    # )
    tabular_model = TabularModel(
        data_config=data_config,
        model_config=model_config,
        optimizer_config=optimizer_config,
        trainer_config=trainer_config,
        # experiment_config=experiment_config,
    )
    tabular_model.fit(
        train=train, validation=val)

    result = tabular_model.evaluate(test)
    print(result)
    test.drop(columns=target_name, inplace=True)
    pred_df = tabular_model.predict(test)
    pred_df.to_csv("output/temp2.csv")
    # tabular_model.save_model("test_save")
    # new_model = TabularModel.load_from_checkpoint("test_save")
    # result = new_model.evaluate(test)









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


"""
import torch
import numpy as np
from torch.functional import norm
# torch.manual_seed(0)
# np.random.seed(0)
# torch.set_deterministic(True)

from sklearn.datasets import fetch_covtype

# from torch.utils import data
from pytorch_tabular.config import (
    DataConfig,
    ExperimentConfig,
    ExperimentRunManager,
    ModelConfig,
    OptimizerConfig,
    TrainerConfig,
)
from pytorch_tabular.models.node.config import NodeConfig
from pytorch_tabular.models.category_embedding.config import (
    CategoryEmbeddingModelConfig,
)
from pytorch_tabular.models.category_embedding.category_embedding_model import (
    CategoryEmbeddingModel,
)
import pandas as pd
from omegaconf import OmegaConf
from pytorch_tabular.tabular_datamodule import TabularDatamodule
from pytorch_tabular.tabular_model import TabularModel
import pytorch_lightning as pl
from sklearn.preprocessing import PowerTransformer
from sklearn.model_selection import train_test_split
from pathlib import Path
import wget


BASE_DIR = Path.home().joinpath('data')
datafile = BASE_DIR.joinpath('covtype.data.gz')
datafile.parent.mkdir(parents=True, exist_ok=True)
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/covtype/covtype.data.gz"
if not datafile.exists():
    wget.download(url, datafile.as_posix())

target_name = ["Covertype"]

cat_col_names = [
    "Wilderness_Area1", "Wilderness_Area2", "Wilderness_Area3",
    "Wilderness_Area4", "Soil_Type1", "Soil_Type2", "Soil_Type3", "Soil_Type4",
    "Soil_Type5", "Soil_Type6", "Soil_Type7", "Soil_Type8", "Soil_Type9",
    "Soil_Type10", "Soil_Type11", "Soil_Type12", "Soil_Type13", "Soil_Type14",
    "Soil_Type15", "Soil_Type16", "Soil_Type17", "Soil_Type18", "Soil_Type19",
    "Soil_Type20", "Soil_Type21", "Soil_Type22", "Soil_Type23", "Soil_Type24",
    "Soil_Type25", "Soil_Type26", "Soil_Type27", "Soil_Type28", "Soil_Type29",
    "Soil_Type30", "Soil_Type31", "Soil_Type32", "Soil_Type33", "Soil_Type34",
    "Soil_Type35", "Soil_Type36", "Soil_Type37", "Soil_Type38", "Soil_Type39",
    "Soil_Type40"
]

num_col_names = [
    "Elevation", "Aspect", "Slope", "Horizontal_Distance_To_Hydrology",
    "Vertical_Distance_To_Hydrology", "Horizontal_Distance_To_Roadways",
    "Hillshade_9am", "Hillshade_Noon", "Hillshade_3pm",
    "Horizontal_Distance_To_Fire_Points"
]

feature_columns = (
    num_col_names + cat_col_names + target_name)

df = pd.read_csv(datafile, header=None, names=feature_columns)
# cat_col_names = []

# num_col_names = [
#     "Elevation", "Aspect"
# ]
# feature_columns = (
#     num_col_names + cat_col_names + target_name)
# df = df.loc[:,feature_columns]
df.head()
train, test = train_test_split(df, random_state=42)
train, val = train_test_split(train, random_state=42)
num_classes = len(set(train[target_name].values.ravel()))

data_config = DataConfig(
    target=target_name,
    continuous_cols=num_col_names,
    categorical_cols=cat_col_names,
    continuous_feature_transform=None,#"quantile_normal",
    normalize_continuous_features=False
)
model_config = CategoryEmbeddingModelConfig(task="classification", metrics=["f1","accuracy"], metrics_params=[{"num_classes":num_classes},{}])
# model_config = NodeConfig(
#     task="classification",
#     depth=4,
#     num_trees=1024,
#     input_dropout=0.0,
#     metrics=["f1", "accuracy"],
#     metrics_params=[{"num_classes": num_classes, "average": "macro"}, {}],
# )
trainer_config = TrainerConfig(gpus=1, fast_dev_run=True)
experiment_config = ExperimentConfig(project_name="PyTorch Tabular Example", 
                                     run_name="node_forest_cov", 
                                     exp_watch="gradients", 
                                     log_target="wandb", 
                                     log_logits=True)
optimizer_config = OptimizerConfig()

# tabular_model = TabularModel(
#     data_config="examples/data_config.yml",
#     model_config="examples/model_config.yml",
#     optimizer_config="examples/optimizer_config.yml",
#     trainer_config="examples/trainer_config.yml",
#     # experiment_config=experiment_config,
# )
tabular_model = TabularModel(
    data_config=data_config,
    model_config=model_config,
    optimizer_config=optimizer_config,
    trainer_config=trainer_config,
    # experiment_config=experiment_config,
)
tabular_model.fit(
    train=train, validation=val)

result = tabular_model.evaluate(test)
print(result)
test.drop(columns=target_name, inplace=True)
pred_df = tabular_model.predict(test)
pred_df.to_csv("output/temp2.csv")
# tabular_model.save_model("test_save")
# new_model = TabularModel.load_from_checkpoint("test_save")
# result = new_model.evaluate(test)

"""