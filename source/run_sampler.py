# pylint: disable=C0321,C0103,E1221,C0301,E1305,E1121,C0302,C0330
# -*- coding: utf-8 -*-
"""

python source/run_sampler.py  run_train --config_name elasticnet  --path_data_train data/input/train/    --path_output data/output/a01_elasticnet/

python source/run_sampler.py  run_train   --n_sample 100  --config_name lightgbm  --path_model_config source/config_model.py  --path_output /data/output/a01_test/     --path_data_train /data/input/train/

"""
import warnings, sys, os, json, importlib, pandas as pd
warnings.filterwarnings('ignore')

####################################################################################################
try   : verbosity = int(json.load(open(os.path.dirname(os.path.abspath(__file__)) + "/../../config.json", mode='r'))['verbosity'])
except Exception as e : verbosity = 2
#raise Exception(f"{e}")

def log(*s):
    print(*s, flush=True)

def log2(*s):
    if verbosity >= 2 : print(*s, flush=True)

def log3(*s):
    if verbosity >= 3 : print(*s, flush=True)

####################################################################################################
#### Add path for python import
sys.path.append( os.path.dirname(os.path.abspath(__file__)) + "/")

#### Root folder analysis
root = os.path.abspath(os.getcwd()).replace("\\", "/") + "/"
print(root)
####################################################################################################

SUPERVISED_MODELS = ['SMOTE', 'SMOTEENN', 'SMOTETomek', 'NearMiss']



####################################################################################################
from util_feature import   load, save_list, load_function_uri, save
from run_preprocess import  preprocess, preprocess_load

def save_features(df, name, path):
    if path is not None :
       os.makedirs( f"{path}/{name}", exist_ok=True)
       df.to_parquet( f"{path}/{name}/features.parquet")


def model_dict_load(model_dict, config_path, config_name, verbose=True):
    """ load the model dict from the python config file.
    :return:
    """
    if model_dict is None :
       log("#### Model Params Dynamic loading  ###############################################")
       model_dict_fun = load_function_uri(uri_name=config_path + "::" + config_name)
       model_dict     = model_dict_fun()   ### params
    log3( model_dict )
    return model_dict


####################################################################################################
##### train    #####################################################################################
def map_model(model_name):
    """
      Get the Class of the object stored in source/models/
    :param model_name:   model_sklearn
    :return: model module

    """
    ##### Custom folder
    if ".py" in model_name :
       path = os.path.parent(model_name)
       sys.path.append(path)
       mod = os.path.basename(model_name)
       modelx = importlib.import_module(mod)
       return modelx

    ##### Local folder
    model_file = model_name.split(":")[0]
    if  'optuna' in model_name : model_file = 'optuna_lightgbm'

    try :
       ##  'models.model_bayesian_pyro'   'model_widedeep'
       mod    = f'models.{model_file}'
       modelx = importlib.import_module(mod)

    except :
        ### All SKLEARN API
        ### ['ElasticNet', 'ElasticNetCV', 'LGBMRegressor', 'LGBMModel', 'TweedieRegressor', 'Ridge']:
       mod    = 'models.model_sklearn'
       modelx = importlib.import_module(mod)
    return modelx


def train(model_dict, dfX, cols_family, post_process_fun):
    """  Train the model using model_dict, save model, save prediction
    :param model_dict:  dict containing params
    :param dfX:  pd.DataFrame
    :param cols_family: dict of list containing column names
    :param post_process_fun:
    :return: dfXtrain , dfXval  DataFrame containing prediction.
    """
    model_pars, compute_pars = model_dict['model_pars'], model_dict['compute_pars']
    data_pars                = model_dict['data_pars']
    model_name, model_path   = model_pars['model_class'], model_dict['global_pars']['path_train_model']
    metric_list              = compute_pars.get('metric_list', [])
    model_file               = model_pars.get('model_file',"model_sampler")

    assert  'cols_model_type2' in data_pars, 'Missing cols_model_type2, split of columns by data type '
    log2(data_pars['cols_model_type2'])


    log("#### Model Input preparation #########################################################")
    log2(dfX.shape)
    dfX    = dfX.sample(frac=1.0)
    itrain = int(0.6 * len(dfX))
    ival   = int(0.8 * len(dfX))
    colsX  = data_pars['cols_model']
    coly   = data_pars['coly']
    log2('Model colsX',colsX)
    log2('Model coly', coly)
    log3('Model column type: ',data_pars['cols_model_type2'])

    data_pars['data_type'] = 'ram'
    data_pars['train'] = {'Xtrain' : dfX[colsX].iloc[:itrain, :],
                          'ytrain' : dfX[coly].iloc[:itrain],
                          'Xtest'  : dfX[colsX].iloc[itrain:ival, :],
                          'ytest'  : dfX[coly].iloc[itrain:ival],

                          'Xval'   : dfX[colsX].iloc[ival:, :],
                          'yval'   : dfX[coly].iloc[ival:],
                          }
    
    data_pars['eval'] = {'X'    : dfX[colsX].iloc[ival:, :],
                         'y'    : dfX[coly].iloc[ival:],
                         }

    log("#### Init, Train ###############################################################")
    # from config_model import map_model    
    if len(model_file) == 0:
        modelx = map_model(model_name)
    else:
        modelx = map_model(model_file +":"+model_name)    
    log2(modelx)
    modelx.reset()
    modelx.init(model_pars, compute_pars=compute_pars)
    modelx.fit(data_pars, compute_pars)


    log("#### Transform ##############################################################")
    if model_name in SUPERVISED_MODELS:
        dfX2, y = modelx.transform((dfX[colsX], dfX[coly]),data_pars=data_pars, compute_pars=compute_pars)
        dfX2    = pd.DataFrame(dfX2, columns = colsX)
    else:
        dfX2    = modelx.transform(dfX[colsX], data_pars=data_pars, compute_pars=compute_pars)
    # dfX2.index = dfX.index

    for coli in dfX2.columns :
       dfX2[coli]            = dfX2[coli].apply(lambda  x : post_process_fun(x) )

    log("Actual    : ",  dfX[colsX])
    log("Prediction: ",  dfX2)

    log("#### Metrics ###############################################################")
    from util_feature import  metrics_eval
    # metrics_test = metrics_eval(metric_list,
    #                             ytrue       = dfX[coly].iloc[ival:],
    #                             ypred       = dfX[coly + '_pred'].iloc[ival:],
    #                             ypred_proba = ypred_proba_val )
    # stats = {'metrics_test' : metrics_test}
    stats = modelx.eval(data_pars=data_pars, compute_pars=compute_pars)
    log(stats)


    log("### Saving model, dfX, columns #############################################")
    log(model_path + "/model.pkl")
    os.makedirs(model_path, exist_ok=True)
    save(colsX, model_path + "/colsX.pkl")
    save(coly,  model_path + "/coly.pkl")
    modelx.save(model_path, stats)


    log("### Reload model,            ###############################################")
    log(modelx.model.model_pars, modelx.model.compute_pars)
    a = load(model_path + "/model.pkl")
    log("Reload model pars", a.model_pars)
    
    return dfX2.iloc[:ival, :].reset_index(), dfX2.iloc[ival:, :].reset_index(), stats


####################################################################################################
############CLI Command ############################################################################
def run_train(config_name, config_path="source/config_model.py", n_sample=5000,
              mode="run_preprocess", model_dict=None, return_mode='file', **kw):
    """
      Configuration of the model is in config_model.py file
    :param config_name:
    :param config_path:
    :param n_sample:
    :return:
    """
    model_dict  = model_dict_load(model_dict, config_path, config_name, verbose=True)

    m           = model_dict['global_pars']
    path_data_train   = m['path_data_train']
    path_train_X      = m.get('path_train_X', path_data_train + "/features.zip") #.zip
    path_train_y      = m.get('path_train_y', path_data_train + "/target.zip")   #.zip

    path_output         = m['path_train_output']
    # path_model          = m.get('path_model',          path_output + "/model/" )
    path_pipeline       = m.get('path_pipeline',       path_output + "/pipeline/" )
    path_features_store = m.get('path_features_store', path_output + '/features_store/' )  #path_data_train replaced with path_output, because preprocessed files are stored there
    path_check_out      = m.get('path_check_out',      path_output + "/check/" )
    log2(path_output)


    log("#### load raw data column family  ###############################################")
    cols_group = model_dict['data_pars']['cols_input_type']  ### Raw data column
    log(cols_group)


    log("#### Preprocess  ################################################################")
    preprocess_pars = model_dict['model_pars']['pre_process_pars']
     
    if mode == "run_preprocess" :
        dfXy, cols      = preprocess(path_train_X, path_train_y,
                                     path_pipeline,    ### path to save preprocessing pipeline
                                     cols_group,       ### dict of column family
                                     n_sample,
                                     preprocess_pars,
                                     path_features_store  ### Store intermediate dataframe
                                     )

    elif mode == "load_preprocess"  :  #### Load existing data
        dfXy, cols      = preprocess_load(path_train_X, path_train_y, path_pipeline, cols_group, n_sample,
                                          preprocess_pars,  path_features_store=path_features_store)


    #### Critical  Part ##################################################################
    log("#### Extract column names  #####################################################")
    ### Flatten Actual column names for Model Input :  label y and Input X (colnum , colcat)
    model_dict['data_pars']['coly']       = cols['coly']
    model_dict['data_pars']['cols_model'] = sum([  cols[colgroup] for colgroup in model_dict['data_pars']['cols_model_group'] ]   , [])


    #### Flatten Col Group by column type : Sparse, continuous, .... (ie Neural Network feed Input
    ## 'coldense' = [ 'colnum' ]     'colsparse' = ['colcat' ]
    model_dict['data_pars']['cols_model_type2'] = {}
    for colg, colg_list in model_dict['data_pars'].get('cols_model_type', {}).items() :
        model_dict['data_pars']['cols_model_type2'][colg] = sum([  cols[colgroup] for colgroup in colg_list ]   , [])



    log("#### Train model: #############################################################")
    log2(str(model_dict)[:1000])
    post_process_fun      = model_dict['model_pars']['post_process_fun']
    dfXy, dfXytest,stats  = train(model_dict, dfXy, cols, post_process_fun)


    log("#### Register model ##########################################################")
    mlflow_pars = model_dict.get('compute_pars', {}).get('mlflow_pars', None)
    if mlflow_pars is not None:
        mlflow_register(dfXy, model_dict, stats, mlflow_pars)


    if return_mode == 'dict' :
        return { 'dfXy' : dfXy, 'dfXytest': dfXytest, 'stats' : stats   }

    else :
        log("#### Export ##################################################################")
        os.makedirs(path_check_out, exist_ok=True)
        dfXy.to_parquet(path_check_out + "/dfX.parquet")  # train input data generate parquet
        dfXytest.to_parquet(path_check_out + "/dfXtest.parquet")  # Test input data  generate parquet
        log("######### Finish #############################################################", )


####################################################################################################
############ Prediction time #######################################################################
def transform(model_name, path_model, dfX, model_dict, task_type='transform'):
    """
    Arguments:
        model_name {[str]} -- [description]
        path_model {[str]} -- [description]
        dfX {[DataFrame]} -- [description]
        cols_family {[dict]} -- [description]

    Returns: ypred
        [numpy.array] -- [vector of prediction]
    """
    modelx = map_model(model_name)
    modelx.reset()
    log(modelx, path_model)
    sys.path.append( root)    #### Needed due to import source error


    log("#### Load model  ############################################")
    log2(path_model + "/model/model.pkl")
    modelx.model = modelx.load(path_model + "/model.pkl")
    colsX        = load(path_model + "/colsX.pkl")   ## column name
    # coly  = load( path_model + "/model/coly.pkl"   )
    assert colsX        is not None, "cannot load colsx, " + path_model
    assert modelx.model is not None, "cannot load modelx, " + path_model
    log("#### modelx\n", modelx.model.model)


    log("### Task Inference  #############################################")
    # dfX1  = dfX.reindex(columns=colsX)   #reindex included

    if task_type == 'encode':
       dfX = modelx.encode(dfX,  data_pars    = model_dict['data_pars'],
                                 compute_pars = model_dict['compute_pars'] )

    elif task_type == 'decode':
       dfX = modelx.encode(dfX,  data_pars    = model_dict['data_pars'],
                                 compute_pars = model_dict['compute_pars'] )
    else :
       dfX = modelx.transform(dfX,  data_pars    = model_dict['data_pars'],
                                    compute_pars = model_dict['compute_pars'] )

    return dfX


####################################################################################################
############CLI Command ############################################################################
from util_feature import load_function_uri, load, load_dataset

def run_transform(config_name, config_path, n_sample=1,
                path_data=None, path_output=None, pars={}, model_dict=None, return_mode=""):

    log("##### Run transform ###############################################################")
    model_dict  = model_dict_load(model_dict, config_path, config_name, verbose=True)
    model_class = model_dict['model_pars']['model_class']

    m                = model_dict['global_pars']
    path_data        = m['path_pred_data']   if path_data   is None else path_data
    path_pipeline    = m['path_pred_pipeline']    #   path_output + "/pipeline/" )
    path_model       = m['path_pred_model']

    model_file       = m.get('model_file', "")  ### New

    path_output      = m['path_pred_output'] if path_output is None else path_output
    log(path_data, path_model, path_output)

    pars = {'cols_group': model_dict['data_pars']['cols_input_type'],
            'pipe_list' : model_dict['model_pars']['pre_process_pars']['pipe_list']}
    

    log("##### Load Preprocess ############################################################")
    from run_preprocess import preprocess_inference   as preprocess
    colid            = load(f'{path_pipeline}/colid.pkl')
    if model_class in SUPERVISED_MODELS:
        path_pred_X      = m.get('path_pred_X', path_data + "/features.zip") #.zip
        path_pred_y      = m.get('path_pred_y', path_data + "/target.zip")   #.zip
        df               = load_dataset(path_pred_X, path_pred_y, colid, n_sample= n_sample)
    else:
        df               = load_dataset(path_data, None, colid, n_sample= n_sample)

    dfX, cols            = preprocess(df, path_pipeline, preprocess_pars=pars)
    coly = cols["coly"]  


    log("#### Extract column names  #######################################################")
    ### Actual column names for Model Input :  label y and Input X (colnum , colcat), remove duplicate names
    model_dict['data_pars']['coly']       = cols['coly']
    model_dict['data_pars']['cols_model'] = list(set(sum([  cols[colgroup] for colgroup in model_dict['data_pars']['cols_model_group'] ]   , []) ))


    ####    Col Group by column type : Sparse, continuous, .... (ie Neural Network feed Input, remove duplicate names
    ####   'coldense' = [ 'colnum' ]     'colsparse' = ['colcat' ]
    model_dict['data_pars']['cols_model_type2'] = {}
    for colg, colg_list in model_dict['data_pars'].get('cols_model_type', {}).items() :
        model_dict['data_pars']['cols_model_type2'][colg] = list(set(sum([  cols[colgroup] for colgroup in colg_list ]   , [])))


    log("############ Task Inference   ###################################################" )
    task_type = model_dict['compute_pars'].get('task_inference', 'transform')
    if model_class in SUPERVISED_MODELS:
        dfXy                = transform(model_file, path_model, (dfX[[c for c in dfX.columns if c not in coly]], df[coly]), model_dict,
                                        task_type= task_type)
    else:
        dfXy                = transform(model_file, path_model, dfX, model_dict,
                                        task_type= task_type)

    post_process_fun        = model_dict['model_pars']['post_process_fun']


    if return_mode == 'dict' :
        return { 'dfXy' : dfXy   }


    else :
        log("#### Export ##################################################################")
        path_check_out      = m.get('path_check_out',      path_output + "/check/" )
        os.makedirs(path_check_out, exist_ok=True)
        dfX.to_parquet(path_check_out + "/dfX.parquet")  # train input data generate parquet
        log("######### Finish #############################################################", )





########################################################################################################################
def mlflow_register(dfXy, model_dict: dict, stats: dict, mlflow_pars:dict ):
    log("#### Using mlflow #########################################################")
    # def register(run_name, params, metrics, signature, model_class, tracking_uri= "sqlite:///local.db"):
    from run_mlflow import register
    from mlflow.models.signature import infer_signature

    train_signature = dfXy[model_dict['data_pars']['cols_model']]
    y_signature     = dfXy[model_dict['data_pars']['coly']]
    signature       = infer_signature(train_signature, y_signature)

    register( run_name    = model_dict['global_pars']['config_name'],
             params       = model_dict['global_pars'],
             metrics      = stats["metrics_test"],
             signature    = signature,
             model_class  = model_dict['model_pars']["model_class"],
             tracking_uri = mlflow_pars.get( 'tracking_db', "sqlite:///mlflow_local.db")
            )



if __name__ == "__main__":
    import fire
    fire.Fire()




