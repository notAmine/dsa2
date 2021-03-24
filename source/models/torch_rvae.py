# pylint: disable=C0321,C0103,C0301,E1305,E1121,C0302,C0330,C0111,W0613,W0611,R1705
# -*- coding: utf-8 -*-
"""

python torch_rvae.py test --nrows 1000


"""
import os, sys,  numpy as np,  pandas as pd, wget, copy
from pathlib import Path
# torch.manual_seed(0)
# np.random.seed(0)
# torch.set_deterministic(True)
# from torch.utils import data
from sklearn.model_selection import train_test_split
import torch


######################################################################################################
sys.path.append( os.path.dirname(os.path.abspath(__file__)) + "/repo/RVAE_MixedTypes/core_models/" )
import main, train_eval_models

root     = os.path.dirname(os.path.abspath(__file__)) 
path_pkg =  root + "/repo/RVAE_MixedTypes/"



import torch
from torch import nn
from torch import optim
from torch.nn import functional as F
from collections import namedtuple
import pandas as pd
import numpy as np
import sys
import json
import os, errno

from model_utils import nll_categ_global, nll_gauss_global
from EmbeddingMul import EmbeddingMul
import parser_arguments
from train_eval_models import training_phase, evaluation_phase, repair_phase
import utils




####################################################################################################
VERBOSE = False

def log(*s):
    print(*s, flush=True)

def log2(*s):
    if VERBOSE :
      print(*s, flush=True)

####################################################################################################
global model, session

def init(*kw, **kwargs):
    global model, session
    model, session = Model(*kw, **kwargs), None


class Model(object):
    def __init__(self, model_pars=None, data_pars=None, compute_pars=None):
        self.model_pars, self.compute_pars, self.data_pars = model_pars, compute_pars, data_pars

        if model_pars is None:
            self.model = None

        else:
            ###############################################################
            dm          = data_pars['cols_model_group_custom']


            self.model = None
            self.guide = None
            self.pred_summary = None  ### All MC summary

            if VERBOSE: log(self.guide, self.model)


def fit(data_pars=None, compute_pars=None, out_pars=None, **kw):
    """
    """
    global model, session
    session = None  # Session type for compute
    cpars          = copy.deepcopy( compute_pars.get("compute_pars", {}))   ## issue with pickle

    # if data_pars is not None :
    Xtrain_tuple, ytrain, Xtest_tuple, ytest = get_dataset(data_pars, task_type="train")








    ###############################################################
    model.model.fit(train=train, validation=val, **cpars)




def encode(Xpred=None, data_pars: dict={}, compute_pars: dict={}, out_pars: dict={}, **kw):
    global model, session




def decode(Xpred=None, data_pars: dict={}, compute_pars: dict={}, out_pars: dict={}, **kw):
    global model, session





def eval(Xpred=None, data_pars: dict={}, compute_pars: dict={}, out_pars: dict={}, **kw):
    global model, session
    """
         Encode + Decode 
    """







def reset():
    global model, session
    model, session = None, None

def save(path=None, info=None):
    """ Custom saving
    """
    global model, session
    import cloudpickle as pickle
    os.makedirs(path + "/model/", exist_ok=True)

    #### Torch part
    model.model.save_model(path + "/model/torch_checkpoint")

    #### Wrapper
    model.model = None   ## prevent issues
    pickle.dump(model,  open(path + "/model/model.pkl", mode='wb')) # , protocol=pickle.HIGHEST_PROTOCOL )
    pickle.dump(info, open(path   + "/model/info.pkl", mode='wb'))  # ,protocol=pickle.HIGHEST_PROTOCOL )


def load_model(path=""):
    global model, session
    import cloudpickle as pickle
    model0 = pickle.load(open(path + '/model/model.pkl', mode='rb'))
         
    model = Model()  # Empty model
    model.model_pars   = model0.model_pars
    model.compute_pars = model0.compute_pars
    model.data_pars    = model0.data_pars

    ### Custom part
    # model.model        = TabularModel.load_from_checkpoint( "ztmp/data/output/torch_tabular/torch_checkpoint")
    model.model        = TabularModel.load_from_checkpoint(  path +"/model/torch_checkpoint")
 
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


# cols_ref_formodel = ['cols_single_group']
cols_ref_formodel = ['colcontinuous', 'colsparse']
def get_dataset_tuple(Xtrain, cols_type_received, cols_ref):
    """  Split into Tuples to feed  Xyuple = (df1, df2, df3) OR single dataframe
    :param Xtrain:
    :param cols_type_received:
    :param cols_ref:
    :return:
    """
    if len(cols_ref) <= 1 :
        return Xtrain

    Xtuple_train = []
    # cols_ref is the reference for types of cols groups (sparse/continuous)
    # This will result in deviding the dataset into many groups of features
    for cols_groupname in cols_ref :
        # Assert the group name is in the cols reference
        assert cols_groupname in cols_type_received, "Error missing colgroup in config data_pars[cols_model_type] "
        cols_i = cols_type_received[cols_groupname]
        # Add the columns of this group to the list
        Xtuple_train.append( Xtrain[cols_i] )

    if len(cols_ref) == 1 :
        return Xtuple_train[0]  ### No tuple
    else :
        return Xtuple_train


def get_dataset(data_pars=None, task_type="train", **kw):
    """
      return tuple of dataframes
    """
    # log(data_pars)
    data_type = data_pars.get('type', 'ram')
    cols_ref  = cols_ref_formodel

    if data_type == "ram":
        # cols_ref_formodel = ['cols_cross_input', 'cols_deep_input', 'cols_deep_input' ]
        ### dict  colgroup ---> list of colname

        cols_type_received     = data_pars.get('cols_model_type2', {} )  ##3 Sparse, Continuous

        if task_type == "predict":
            d = data_pars[task_type]
            Xtrain       = d["X"]
            Xtuple_train = get_dataset_tuple(Xtrain, cols_type_received, cols_ref)
            return Xtuple_train

        if task_type == "eval":
            d = data_pars[task_type]
            Xtrain, ytrain  = d["X"], d["y"]
            Xtuple_train    = get_dataset_tuple(Xtrain, cols_type_received, cols_ref)
            return Xtuple_train, ytrain

        if task_type == "train":
            d = data_pars[task_type]
            Xtrain, ytrain, Xtest, ytest  = d["Xtrain"], d["ytrain"], d["Xtest"], d["ytest"]

            ### dict  colgroup ---> list of df
            Xtuple_train = get_dataset_tuple(Xtrain, cols_type_received, cols_ref)
            Xtuple_test  = get_dataset_tuple(Xtest, cols_type_received, cols_ref)
            log2("Xtuple_train", Xtuple_train)

            return Xtuple_train, ytrain, Xtuple_test, ytest


    elif data_type == "file":
        raise Exception(f' {data_type} data_type Not implemented ')

    raise Exception(f' Requires  Xtrain", "Xtest", "ytrain", "ytest" ')


####################################################################################################
############ Test  #################################################################################
def test_dataset_1(nrows=1000):
    # Dense features
    colnum = []

    # Sparse features
    colcat = [  ]



def test(nrows=1000):
    """
        nrows : take first nrows from dataset
    """
    global model, session
    df, colnum, colcat, coly = test_dataset_covtype()

    #### Matching Big dict  ##################################################    
    X = df
    y = df[coly].astype('uint8')
    log('y', np.sum(y[y==1]) )

    # Split the df into train/test subsets
    X_train_full, X_test, y_train_full, y_test = train_test_split(X, y, test_size=0.05, random_state=2021, stratify=y)
    X_train, X_valid, y_train, y_valid         = train_test_split(X_train_full, y_train_full, random_state=2021, stratify=y_train_full)
    num_classes = len(set(y_train_full[coly].values.ravel()))
    log(X_train)





    m = {'model_pars': {
        ### LightGBM API model   #######################################
        # Specify the ModelConfig for pytorch_tabular
        'model_class':  "torch_tabular.py::CategoryEmbeddingModelConfig"
        
        # Type of target prediction, evaluation metrics
        ,'model_pars' : { 
                        # 'task': "classification",
                        # 'metrics' : ["f1","accuracy"],
                        # 'metrics_params' : [{"num_classes":num_classes},{}]
                        }  

        , 'post_process_fun' : post_process_fun   ### After prediction  ##########################################
        , 'pre_process_pars' : {'y_norm_fun' :  pre_process_fun ,  ### Before training  ##########################

        ### Pipeline for data processing ##############################
        'pipe_list': [  #### coly target prorcessing
        {'uri': 'source/prepro.py::pd_coly',                 'pars': {}, 'cols_family': 'coly',       'cols_out': 'coly',           'type': 'coly'         },

        {'uri': 'source/prepro.py::pd_colnum_bin',           'pars': {}, 'cols_family': 'colnum',     'cols_out': 'colnum_bin',     'type': ''             },
        {'uri': 'source/prepro.py::pd_colnum_binto_onehot',  'pars': {}, 'cols_family': 'colnum_bin', 'cols_out': 'colnum_onehot',  'type': ''             },

        #### catcol INTO integer,   colcat into OneHot
        {'uri': 'source/prepro.py::pd_colcat_bin',           'pars': {}, 'cols_family': 'colcat',     'cols_out': 'colcat_bin',     'type': ''             },
        {'uri': 'source/prepro.py::pd_colcat_to_onehot',     'pars': {}, 'cols_family': 'colcat_bin', 'cols_out': 'colcat_onehot',  'type': ''             },

        ],
            }
        },

    'compute_pars': { 'metric_list': ['accuracy_score','average_precision_score']
                    },

    'data_pars': { 'n_sample' : n_sample,
        'download_pars' : None,
        'cols_input_type' : cols_input_type_1,
        ### family of columns for MODEL  #########################################################
        'cols_model_group': [ 'colnum_bin',   'colcat_bin',
                            ]

        ,'cols_model_group_custom' :  { 'colnum' : colnum,
                                        'colcat' : colcat,
                                        'coly' : coly
                            }
        ###################################################  
        ,'train': {'Xtrain': X_train, 'ytrain': y_train,
                   'Xtest': X_valid,  'ytest':  y_valid},
                'eval': {'X': X_valid,  'y': y_valid},
                'predict': {'X': X_valid}

        ### Filter data rows   ##################################################################
        ,'filter_pars': { 'ymax' : 2 ,'ymin' : -1 },

        
        ### Added continuous & sparse features groups ###
        'cols_model_type2': {
            'colcontinuous':   colnum ,
            'colsparse' : colcat, 
        },
        }
    }


    ll = [
        ('torch_tabular.py::CategoryEmbeddingModelConfig', 
            {   'task': "classification",
                'metrics' : ["f1","accuracy"],
                'metrics_params' : [{"num_classes":num_classes},{}]
            }  
        ),

    
    ]
    for cfg in ll:
        log("******************************************** New Model ********************************************")
        log(f"******************************************** {cfg[0]} ********************************************")
        reset()
        # Set the ModelConfig
        m['model_pars']['model_class'] = cfg[0]
        m['model_pars']['model_pars']  = {**m['model_pars']['model_pars'] , **cfg[1] }

        log('Setup model..')
        model = Model(model_pars=m['model_pars'], data_pars=m['data_pars'], compute_pars= m['compute_pars'] )

        log('\n\nTraining the model..')
        fit(data_pars=m['data_pars'], compute_pars= m['compute_pars'], out_pars=None)
        log('Training completed!\n\n')

        log('Predict data..')
        ypred, ypred_proba = predict(Xpred=None, data_pars=m['data_pars'], compute_pars=m['compute_pars'])
        log(f'Top 5 y_pred: {np.squeeze(ypred)[:5]}')

        if cfg != "torch_tabular.py::NodeConfig":
            log('Saving model..')
            save(path= "ztmp/data/output/torch_tabular")
            #  os.path.join(root, 'data\\output\\torch_tabular\\model'))

            log('Load model..')
            model, session = load_model(path="ztmp/data/output/torch_tabular")
            #os.path.join(root, 'data\\output\\torch_tabular\\model'))
        else:
            log('\n*** !!! Saving Bug in pytorch_tabular for NodeConfig !!! ***\n')
            
        log('Model architecture:')
        log(model.model)

        log('Model config:')
        log(model.model.config._config_name)
        reset()


def test2(nrow=10000):
    """
       python source/models/torch_tabular.py test

    """
    global model, session


if __name__ == "__main__":
    import fire
    fire.Fire()
    # test()

















#!/usr/bin/env python3
class RVAE(nn.Module):
    def __init__(
        self, 
        data_folder, 
        args):

        super(RVAE, self).__init__()
        # NOTE: for feat_select, (col_name, col_type, feat_size) in enumerate(dataset_obj.feat_info)
        self.data_folder = data_folder
        self.args = args
        self.last_epoch = None
        # Load dataset and get dataset_obj
        dataset_obj = self._get_dataset_obj() 

        self.dataset_obj = dataset_obj

        self.size_input = len(dataset_obj.cat_cols)*self.args.embedding_size + len(dataset_obj.num_cols)
        self.size_output = len(dataset_obj.cat_cols) + len(dataset_obj.num_cols) # 2*

        ## Encoder Params

        # define a different embedding matrix for each feature
        if (dataset_obj.dataset_type == "image") and (not dataset_obj.cat_cols):
            self.feat_embedd = nn.ModuleList([])
        else:
            self.feat_embedd = nn.ModuleList([nn.Embedding(c_size, self.args.embedding_size, max_norm=1)
                                             for _, col_type, c_size in dataset_obj.feat_info
                                             if col_type=="categ"])

        self.fc1 = nn.Linear(self.size_input, self.args.layer_size)
        self.fc21 = nn.Linear(self.args.layer_size, self.args.latent_dim)
        self.fc22 = nn.Linear(self.args.layer_size, self.args.latent_dim)

        if self.args.AVI:
            self.qw_fc1 = nn.Linear(self.size_input, self.args.layer_size)
            self.qw_fc2 = nn.Linear(self.args.layer_size, len(dataset_obj.feat_info))

        ## Decoder Params

        self.fc3 = nn.Linear(self.args.latent_dim, self.args.layer_size)

        if dataset_obj.dataset_type == "image" and (not dataset_obj.cat_cols):
            self.out_cat_linears = nn.Linear(self.args.layer_size, self.size_output)
        else:
            self.out_cat_linears = nn.ModuleList([nn.Linear(self.args.layer_size, c_size) if col_type=="categ"
                                                 else nn.Linear(self.args.layer_size, c_size) # 2*
                                                 for _, col_type, c_size in dataset_obj.feat_info])

        ## Log variance of the decoder for real attributes
        if dataset_obj.dataset_type == "image" and (not dataset_obj.cat_cols):
            self.logvar_x = nn.Parameter(torch.zeros(1).float())
        else:
            if dataset_obj.num_cols:
                self.logvar_x = nn.Parameter(torch.zeros(1,len(dataset_obj.num_cols)).float())
            else:
                self.logvar_x = []

        ## Other

        if self.args.activation == 'relu':
            self.activ = nn.ReLU()
        elif self.args.activation == 'hardtanh':
            self.activ = nn.Hardtanh()

        self.logSoftmax = nn.LogSoftmax(dim=1)
        self.sigmoid = nn.Sigmoid()

        # define encoder / decoder easy access parameter list
        encoder_list = [self.fc1, self.fc21, self.fc22]
        self.encoder_mod = nn.ModuleList(encoder_list)
        if self.args.AVI:
            encoder_list = [self.qw_fc1, self.qw_fc2]
            self.encoder_mod.extend(encoder_list)
        if self.feat_embedd:
            self.encoder_mod.append(self.feat_embedd)

        self.encoder_param_list = nn.ParameterList(self.encoder_mod.parameters())

        decoder_list = [self.fc3, self.out_cat_linears]
        self.decoder_mod = nn.ModuleList(decoder_list)
        self.decoder_param_list = nn.ParameterList(self.decoder_mod.parameters())
        if len(self.logvar_x):
            self.decoder_param_list.append(self.logvar_x)
    
    def _get_dataset_obj(self):

        self.train_loader, \
        self.X_train, \
        self.target_errors_train, \
        dataset_obj, \
        self.attributes = utils.load_data(self.data_folder, self.args.batch_size,
                                        is_train=True,
                                        get_data_idxs=False)

        self.test_loader, self.X_test, self.target_errors_test, _, _ = utils.load_data(
            self.args.data_folder, 
            self.args.batch_size, 
            is_train=False
        )
        # -- clean versions for evaluation
        _, self.X_train_clean, _, _, _ = utils.load_data(
            self.args.data_folder, 
            self.args.batch_size,
            is_train=True, 
            is_clean=True, 
            stdize_dirty=True
        )
        _, self.X_test_clean, _, _, _ = utils.load_data(
            self.args.data_folder, 
            self.args.batch_size, 
            is_train=False,
            is_clean=True, 
            stdize_dirty=True
        )
        return dataset_obj


    def fit(self):

        optimizer = optim.Adam(
            filter(
                lambda p: p.requires_grad, 
                self.parameters()
            ),
            lr=self.args.lr, 
            weight_decay=self.args.l2_reg
        )  # excludes frozen params / layers

        # Run epochs
        for epoch in range(1, self.args.number_epochs + 1):

            # Training Phase
            self.last_epoch = epoch
            _train_loader, _dataset_obj = self.train_loader, self.dataset_obj
            training_phase(self, optimizer, _train_loader, self.args, epoch)

    
    def save(self):
        # structs for saving data
        losses_save = {"train":{},"test":{}, "train_per_feature":{}, "test_per_feature":{}}
        # create folder for saving experiment data (if necessary)
        folder_output = self.args.output_folder + "/" + self.args.outlier_model

        ### Train Data
        self._save_to_csv(
            self.X_train, 
            self.X_train_clean, 
            self.target_errors_train, 
            self.attributes, 
            losses_save,
            self.dataset_obj, 
            folder_output, 
            self.args, 
            self.last_epoch, 
            mode='train'
        )


        ### Test Data
        self._save_to_csv(
            self.X_test, 
            self.X_test_clean, 
            self.target_errors_test, 
            self.attributes, 
            losses_save,
            self.dataset_obj, 
            folder_output, 
            self.args, 
            self.last_epoch, 
            mode='test'
        )


        # save model parameters
        self.cpu()
        torch.save(self.state_dict(), folder_output + "/model_params.pth")

        # save to .json file the args that were used for running the model
        with open(folder_output + "/args_run.json", "w") as outfile:
            # json.dump(vars(self.args), outfile, indent=4, sort_keys=True)
            json.dump(dir(self.args), outfile, indent=4, sort_keys=True)

    def _save_to_csv(self, X_data, X_data_clean, target_errors, attributes, losses_save,
                dataset_obj, folder_output, args, epoch, mode='train'):

        """ This method performs all operations needed to save the data to csv """

        #Create saving folderes
        try:
            os.makedirs(folder_output)
        except OSError as e:
            if e.errno != errno.EEXIST:
                raise


        ### Evaluate model
        _, metric_ret = evaluation_phase(self, X_data, dataset_obj, args, epoch)

        clean_loss_ret = repair_phase(self, X_data, X_data_clean, dataset_obj, args, target_errors, mode, epoch)


        ## calc cell metrics
        auc_cell_nll, auc_vec_nll, avpr_cell_nll, avpr_vec_nll = utils.cell_metrics(target_errors, metric_ret['nll_score'], weights=False)
        if args.outlier_model == "RVAE":
            auc_cell_pi, auc_vec_pi, avpr_cell_pi, avpr_vec_pi = utils.cell_metrics(target_errors, metric_ret['pi_score'], weights=True)
        else:
            auc_cell_pi, auc_vec_pi, avpr_cell_pi, avpr_vec_pi = -10, np.zeros(len(attributes))*-10, -10, np.zeros(len(attributes))*-10


        # store AVPR for features (cell only)
        df_avpr_feat_cell = pd.DataFrame([], index=['AVPR_nll', 'AVPR_pi'], columns=attributes)
        df_avpr_feat_cell.loc['AVPR_nll'] = avpr_vec_nll
        df_avpr_feat_cell.loc['AVPR_pi'] = avpr_vec_pi
        df_avpr_feat_cell.to_csv(folder_output + "/" + mode + "_avpr_features.csv")

        # store AUC for features (cell only)
        df_auc_feat_cell = pd.DataFrame([], index=['AUC_nll', 'AUC_pi'], columns=attributes)
        df_auc_feat_cell.loc['AUC_nll'] = auc_vec_nll
        df_auc_feat_cell.loc['AUC_pi'] = auc_vec_pi
        df_auc_feat_cell.to_csv(folder_output + "/" + mode + "_auc_features.csv")

        ### Store data from Epochs
        columns = ['Avg. AVI Loss', 'Avg. AVI NLL', 'Avg. AVI KLD_Z', 'Avg. AVI KLD_W',
                'Avg. SEQ Loss', 'Avg. SEQ NLL', 'Avg. SEQ KLD_Z', 'Avg. SEQ KLD_W',
                'Avg. Loss -- p(x_dirty | x_dirty) on all', 'Avg. NLL -- p(x_dirty | x_dirty) on all', 'Avg. KLD_Z -- p(x_dirty | x_dirty) on all', 'Avg. KLD_W -- p(x_dirty | x_dirty) on all',
                'Avg. Loss -- p(x_clean | x_dirty) on dirty pos', 'Avg. NLL -- p(x_clean | x_dirty) on dirty pos', 'Avg. KLD_Z -- p(x_clean | x_dirty) on dirty pos', 'Avg. KLD_W -- p(x_clean | x_dirty) on dirty pos',
                'Avg. Loss -- p(x_clean | x_dirty) on clean pos', 'Avg. NLL -- p(x_clean | x_dirty) on clean pos', 'Avg. KLD_Z -- p(x_clean | x_dirty) on clean pos', 'Avg. KLD_W -- p(x_clean | x_dirty) on clean pos',
                'Avg. Loss -- p(x_clean | x_dirty) on all', 'Avg. NLL -- p(x_clean | x_dirty) on all', 'Avg. KLD_Z -- p(x_clean | x_dirty) on all', 'Avg. KLD_W -- p(x_clean | x_dirty) on all',
                'W Norm Convergence', 'AUC Cell nll score', 'AVPR Cell nll score', 'AUC Row nll score', 'AVPR Row nll score',
                'AUC Cell pi score', 'AVPR Cell pi score', 'AUC Row pi score', 'AVPR Row pi score',
                'Error lower-bound on dirty pos', 'Error upper-bound on dirty pos', 'Error repair on dirty pos', 'Error repair on clean pos']

        df_out = pd.DataFrame.from_dict(losses_save[mode], orient="index",
                                        columns=columns)
        df_out.index.name = "Epochs"
        df_out.to_csv(folder_output + "/" + mode + "_epochs_data.csv")

        ### Store errors per feature

        df_errors_repair = pd.DataFrame([], index=['error_lowerbound_dirtycells','error_repair_dirtycells',
                'error_upperbound_dirtycells','error_repair_cleancells'], columns=attributes)
        df_errors_repair.loc['error_lowerbound_dirtycells'] = clean_loss_ret['errors_per_feature'][0].cpu()
        df_errors_repair.loc['error_repair_dirtycells'] = clean_loss_ret['errors_per_feature'][1].cpu()
        df_errors_repair.loc['error_upperbound_dirtycells'] = clean_loss_ret['errors_per_feature'][2].cpu()
        df_errors_repair.loc['error_repair_cleancells'] = clean_loss_ret['errors_per_feature'][3].cpu()
        df_errors_repair.to_csv(folder_output + "/" + mode + "_error_repair_features.csv")


    def get_inputs(self, x_data, one_hot_categ=False, masking=False, drop_mask=[], in_aux_samples=[]):

        """
            drop_mask: (N,D) defines which entries are to be zeroed-out
        """

        if not masking:
            drop_mask = torch.ones(x_data.shape, device=x_data.device)

        if not isinstance(in_aux_samples, list):
            aux_samples_on = True
        else:
            aux_samples_on = False

        if self.dataset_obj.dataset_type == "image" and (not self.dataset_obj.cat_cols):
            # image data, hence real
            return x_data*drop_mask

        else:
            # mixed data, or just real or just categ
            input_list = []
            cursor_embed = 0
            start = 0

            for feat_idx, ( _, col_type, feat_size ) in enumerate(self.dataset_obj.feat_info):

                if one_hot_categ:
                    if col_type == "categ": # categorical (uses embeddings)
                        func_embedd = EmbeddingMul(self.args.embedding_size, x_data.device)
                        func_embedd.requires_grad = x_data.requires_grad
                        categ_val = func_embedd(x_data[:,start:(start + feat_size)].view(1,x_data.shape[0],-1),
                                    self.feat_embedd[cursor_embed].weight,-1, max_norm=1, one_hot_input=True)
                        input_list.append(categ_val.view(x_data.shape[0],-1)*drop_mask[:,feat_idx].view(-1,1))

                        start += feat_size
                        cursor_embed += 1

                    elif col_type == "real": # numerical
                        input_list.append((x_data[:,start]*drop_mask[:,feat_idx]).view(-1,1))
                        start += 1

                else:
                    if col_type == "categ": # categorical (uses embeddings)
                        if aux_samples_on:
                            aux_categ = self.feat_embedd[cursor_embed](x_data[:,feat_idx].long())*drop_mask[:,feat_idx].view(-1,1) \
                                + (1.-drop_mask[:,feat_idx].view(-1,1))*self.feat_embedd[cursor_embed](in_aux_samples[:,feat_idx].long())
                        else:
                            aux_categ = self.feat_embedd[cursor_embed](x_data[:,feat_idx].long())*drop_mask[:,feat_idx].view(-1,1)
                        input_list.append(aux_categ)
                        cursor_embed += 1

                    elif col_type == "real": # numerical
                        if aux_samples_on:
                            input_list.append((x_data[:,feat_idx]*drop_mask[:,feat_idx]).view(-1,1) \
                                + ((1.-drop_mask[:,feat_idx])*in_aux_samples[:,feat_idx]).view(-1,1) )
                        else:
                            input_list.append((x_data[:,feat_idx]*drop_mask[:,feat_idx]).view(-1,1))

            return torch.cat(input_list, 1)



    def encode(self, x_data, one_hot_categ=False, masking=False, drop_mask=[], in_aux_samples=[]):

        q_params = dict()

        input_values = self.get_inputs(x_data, one_hot_categ, masking, drop_mask, in_aux_samples)

        fc1_out = self.fc1(input_values)

        h1_qz = self.activ(fc1_out)

        q_params['z'] = {'mu': self.fc21(h1_qz), 'logvar': self.fc22(h1_qz)}

        if self.args.AVI:

            qw_fc1_out = self.qw_fc1(input_values)

            h1_qw = self.activ(qw_fc1_out)

            q_params['w'] = {'logit_pi': self.qw_fc2(h1_qw)}

        return q_params

    def sample_normal(self, q_params_z, eps=None):

        if self.training:

            if eps is None:
                eps = torch.randn_like(q_params_z['mu'])

            std = q_params_z['logvar'].mul(0.5).exp_()

            return eps.mul(std).add_(q_params_z['mu'])

        else:
            return q_params_z['mu']

    def reparameterize(self, q_params, eps_samples=None):

        q_samples = dict()

        q_samples['z'] = self.sample_normal(q_params['z'], eps_samples)

        return q_samples


    def decode(self, z):

        p_params = dict()

        h3 = self.activ(self.fc3(z))

        if self.dataset_obj.dataset_type == 'image' and (not self.dataset_obj.cat_cols):

            # tensor with dims (batch_size, self.size_output)
            p_params['x'] = self.out_cat_linears(h3)
            p_params['logvar_x'] = self.logvar_x.clamp(-3,3)

        else:
            out_cat_list = []

            for feat_idx, out_cat_layer in enumerate(self.out_cat_linears):

                if self.dataset_obj.feat_info[feat_idx][1] == "categ": # coltype check
                    out_cat_list.append(self.logSoftmax(out_cat_layer(h3)))

                elif self.dataset_obj.feat_info[feat_idx][1] == "real":
                    out_cat_list.append(out_cat_layer(h3))

            # tensor with dims (batch_size, self.size_output)
            p_params['x'] = torch.cat(out_cat_list, 1)

            if self.dataset_obj.num_cols:
                p_params['logvar_x'] = self.logvar_x.clamp(-3,3)

        return p_params


    def forward(self, x_data, n_epoch=None, one_hot_categ=False, masking=False, drop_mask=[], in_aux_samples=[]):

        q_params = self.encode(x_data, one_hot_categ, masking, drop_mask, in_aux_samples)
        q_samples = self.reparameterize(q_params)

        return self.decode(q_samples['z']), q_params, q_samples


    def loss_function(self, input_data, p_params, q_params, q_samples, clean_comp_only=False, data_eval_clean=False):

        """ ELBO: reconstruction loss for each variable + KL div losses summed over elements of a batch """

        dtype_float = torch.cuda.FloatTensor if self.args.cuda_on else torch.FloatTensor
        nll_val = torch.zeros(1).type(dtype_float)

        if self.dataset_obj.dataset_type == 'image' and (not self.dataset_obj.cat_cols):
            # image datasets, large number of features (so vectorize loss and pi calc.)
            pi_feat = torch.sigmoid(q_params['w']['logit_pi']).clamp(1e-6, 1-1e-6)

            if clean_comp_only and data_eval_clean:
                pi_feat = torch.ones_like(q_params['w']['logit_pi'])

            nll_val = nll_gauss_global(p_params['x'],
                                       input_data,
                                       p_params['logvar_x'], isRobust=True,
                                       std_0_scale=self.args.std_gauss_nll,
                                       w=pi_feat, isClean=clean_comp_only,
                                       shape_feats=[len(self.dataset_obj.num_cols)]).sum()

        else:
            # mixed datasets, or just categorical / continuous with medium number of features
            start = 0
            cursor_num_feat = 0

            for feat_select, (_, col_type, feat_size) in enumerate(self.dataset_obj.feat_info):


                pi_feat = torch.sigmoid(q_params['w']['logit_pi'][:,feat_select]).clamp(1e-6, 1-1e-6)

                if clean_comp_only and data_eval_clean:
                    pi_feat = torch.ones_like(q_params['w']['logit_pi'][:,feat_select])

                # compute NLL
                if col_type == 'categ':

                    nll_val += nll_categ_global(p_params['x'][:,start:(start + feat_size)],
                                                input_data[:,feat_select].long(), feat_size, isRobust=True,
                                                w=pi_feat, isClean=clean_comp_only).sum()

                    start += feat_size

                elif col_type == 'real':

                    nll_val += nll_gauss_global(p_params['x'][:,start:(start + 1)], # 2
                                                input_data[:,feat_select],
                                                p_params['logvar_x'][:,cursor_num_feat], isRobust=True,
                                                w=pi_feat, isClean=clean_comp_only, 
                                                std_0_scale=self.args.std_gauss_nll).sum()

                    start += 1 # 2
                    cursor_num_feat +=1


        # kld regularizer on the latent space
        z_kld = -0.5 * torch.sum(1 + q_params['z']['logvar'] - q_params['z']['mu'].pow(2) - q_params['z']['logvar'].exp())

        # prior on clean cells (higher values means more likely to be clean)
        prior_sig = torch.tensor(self.args.alpha_prior).type(dtype_float)

        # kld regularized on the weights
        pi_mtx = torch.sigmoid(q_params['w']['logit_pi']).clamp(1e-6, 1-1e-6)
        w_kld = torch.sum(pi_mtx * torch.log(pi_mtx / prior_sig) + (1-pi_mtx) * torch.log((1-pi_mtx) / (1-prior_sig)))

        loss_ret = nll_val + z_kld if clean_comp_only else nll_val + z_kld + w_kld

        return loss_ret, nll_val, z_kld, w_kld 

def test():
    args={
        "AVI":False, 
        "activation":'relu', 
        "alpha_prior":0.95, 
        "batch_size":150, 
        "cuda_on":False, 
        "data_folder":'../data_simple/Wine/gaussian_m0s5_categorical_alpha0.0/5pc_rows_20pc_cols_run_1/',
        "embedding_size":50, 
        "inference_type":'vae', 
        "is_one_hot":False, 
        "l2_reg":0.0, 
        "latent_dim":20, 
        "layer_size":400, 
        "load_model":False, 
        "load_model_path":None, 
        "log_interval":50, 
        "lr":0.001, 
        "number_epochs":2, 
        "outlier_model":'RVAE', 
        "output_folder":'outputs_experiments_i/Wine/gaussian_m0s5_categorical_alpha0.0/5pc_rows_20pc_cols_run_1/RVAE_CVI', 
        "save_on":True, 
        "seqvae_bprop":False, 
        "seqvae_steps":4, 
        "seqvae_two_stage":False, 
        "std_gauss_nll":2.0,
        "steps_2stage":4, 
        "verbose_metrics_epoch":True, 
        "verbose_metrics_feature_epoch":False
    }
    args = namedtuple("args", args.keys())(*args.values())

    model = RVAE(
        data_folder=args.data_folder,
        args=args
    )
    model.fit()
    model.save()
    
if __name__ == "__main__":
    # args = parser_arguments.getArgs(sys.argv[1:])
    import fire
    fire.Fire()