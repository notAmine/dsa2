# pylint: disable=C0321,C0103,E1221,C0301,E1305,E1121,C0302,C0330
# -*- coding: utf-8 -*-
"""
https://www.kaggle.com/tapioca/multiclass-lightgbm

https://medium.com/@nitin9809/lightgbm-binary-classification-multi-class-classification-regression-using-python-4f22032b36a2



All in one file config
!  python multiclass_classifier.py  train
!  python multiclass_classifier.py  check
!  python multiclass_classifier.py  predict
"""
import warnings, copy
warnings.filterwarnings('ignore')
import os, sys
import pandas as pd

###################################################################################
from source import util_feature


###### Path ########################################################################
print( os.getcwd())
root = os.path.abspath(os.getcwd()).replace("\\", "/") + "/"
print(root)

dir_data  = os.path.abspath( root + "/data/" ) + "/"
dir_data  = dir_data.replace("\\", "/")
print(dir_data)


####################################################################################
config_file  = f"multiclass_classifier.py"
data_name    = f"multiclass_wine"     ### in data/input/



config_name  = 'multiclass_lightgbm'
n_sample     =  1000


colid   = ''
coly    = ''
colcat  = []
colnum  = []



####################################################################################
##### Params########################################################################
def multiclass_lightgbm(path_model_out="") :
    """
       multiclass
    """
    global path_config_model, path_model, path_data_train, path_data_test, path_output_pred, n_sample,model_name

    config_name       = 'multiclass_lightgbm'
    model_name        = 'LGBMClassifier'

    path_config_model = root + f"/{config_file}"
    path_model        = f'data/output/{data_name}/a01_{model_name}/'
    path_data_train   = f'data/input/{data_name}/train/'
    path_data_test    = f'data/input/{data_name}/test/'
    path_output_pred  = f'/data/output/{data_name}/pred_a01_{config_name}/'

    n_sample    = 1000


    def post_process_fun(y):
        ### After prediction is done
        return  y.astype('int')

    def pre_process_fun(y):
        ### Before the prediction is done
        return  y.astype('int')

    model_dict = {'model_pars': {
        'model_path'       : path_model_out

        ### LightGBM API model       ###################################
        ,'config_model_name': model_name    ## ACTUAL Class name for model_sklearn.py
        ,'model_pars'       : {'objective': 'binary',
                                'learning_rate':0.03,'boosting_type':'gbdt'


                               }

        ### After prediction  ##########################################
        , 'post_process_fun' : post_process_fun
        , 'pre_process_pars' : {'y_norm_fun' :  None ,
                                
                                ### Pipeline for data processing.
                               'pipe_list'  : [ 'filter',     ### Fitler the data
                                                'label',      ### Normalize the label
                                                'dfnum_bin',
                                                'dfnum_hot',
                                                'dfcat_bin',
                                                'dfcat_hot',
                                                'dfcross_hot', ]
                               }
        },
      'compute_pars': { 'metric_list': ['accuracy_score','average_precision_score']
                      },

      'data_pars': {
          'cols_input_type' : {
                     "coly"   :   coly
                    ,"colid"  :   colid
                    ,"colcat" :   colcat
                    ,"colnum" :   colnum
                    ,"coltext" :  []
                    ,"coldate" :  []
                    ,"colcross" : colcross
                   },

          ### used for the model input
          # cols['cols_model'] = cols["colnum"] + cols["colcat_bin"]  # + cols[ "colcross_onehot"]
          'cols_model_group': [ 'colnum', 'colcat_bin']


          ### Actual column namaes to be filled automatically
         ,'cols_model':       []      # cols['colcat_model'],
         ,'coly':             []      # cols['coly']


          ### Filter data rows
         ,'filter_pars': { 'ymax' : 2 ,'ymin' : -1 }

         }

     ,'global_pars' : {}
      }

    lvars = [ 'config_name', 'model_name', 'path_config_model', 'path_model', 'path_data_train', 
              'path_data_test', 'path_output_pred', 'n_sample'
            ]
    for t in lvars:
      model_dict['global_pars'][t] = globals()[t] 


    return model_dict







####################################################################################################
########## Init variable ###########################################################################
globals()[config_name]()




###################################################################################
########## Preprocess #############################################################
def preprocess():
    from source import run_preprocess

    run_preprocess.run_preprocess(model_name        =  config_name, 
                                  path_data         =  path_data_train, 
                                  path_output       =  path_model, 
                                  path_config_model =  path_config_model, 
                                  n_sample          =  n_sample,
                                  mode              =  'run_preprocess')

############################################################################
########## Train ###########################################################
def train():
    from source import run_train

    run_train.run_train(config_model_name =  config_name,
                        path_data         =  path_data_train,
                        path_output       =  path_model,
                        path_config_model =  path_config_model , n_sample = n_sample)


###################################################################################
######### Check model #############################################################
def check():
    from source import run_train
    run_train.run_check(path_output =  path_model,
                        scoring     =  'accuracy' )



    #! python source/run_inference.py  run_predict  --config_model_name  LGBMRegressor  --n_sample 1000   --path_model /data/output/a01_lightgbm_huber/    --path_output /data/output/pred_a01_lightgbm_huber/    --path_data /data/input/train/



########################################################################################
####### Inference ######################################################################
def predict():
    from source import run_inference
    run_inference.run_predict(model_name,
                              path_model  = path_model,
                              path_data   = path_data_test,
                              path_output = path_output_pred,
                              n_sample    = n_sample)


def run_all():
    preprocess()
    train()
    check()
    predict()



###########################################################################################################
###########################################################################################################
"""
python  multiclass_classifier.py  data_profile
python  multiclass_classifier.py  preprocess
python  multiclass_classifier.py  train
python  multiclass_classifier.py  check
python  multiclass_classifier.py  predict
python  multiclass_classifier.py  run_all
"""
if __name__ == "__main__":
    import fire
    fire.Fire()
    
