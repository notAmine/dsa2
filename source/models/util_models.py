# pylint: disable=C0321,C0103,C0301,E1305,E1121,C0302,C0330,C0111,W0613,W0611,R1705
# -*- coding: utf-8 -*-
"""

"""
import logging, os, pandas as pd
from sklearn.model_selection import train_test_split

####################################################################################################
VERBOSE = False

def log(*s):
    print(*s, flush=True)



####################################################################################################
def test_dataset_classifier_covtype(nrows=500):
    import wget
    # Dense features
    colnum = ["Elevation", "Aspect", "Slope", "Horizontal_Distance_To_Hydrology",]

    # Sparse features
    colcat = ["Wilderness_Area1",  "Wilderness_Area2", "Wilderness_Area3",
        "Wilderness_Area4",  "Soil_Type1",  "Soil_Type2",  "Soil_Type3",
        "Soil_Type4",  "Soil_Type5",  "Soil_Type6",  "Soil_Type7",  "Soil_Type8",  "Soil_Type9",  ]

    # Target column
    coly        = ["Covertype"]

    log("start")
    global model, session

    root     = os.path.join(os.getcwd() ,"ztmp")
    BASE_DIR = Path.home().joinpath( root, 'data/input/covtype/')
    datafile = BASE_DIR.joinpath('covtype.data.gz')
    datafile.parent.mkdir(parents=True, exist_ok=True)
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/covtype/covtype.data.gz"

    # Download the dataset in case it's missing
    if not datafile.exists():
        wget.download(url, datafile.as_posix())

    # Read nrows of only the given columns
    feature_columns = colnum + colcat + coly
    df = pd.read_csv(datafile, header=None, names=feature_columns, nrows=nrows)
    #### Matching Big dict  ##################################################
    # X = df
    # y = df[coly].astype('uint8')
    return df, colnum, colcat, coly



def test_dataset_regress_fake(nrows=500):
    from sklearn import datasets as sklearn_datasets
    coly = 'y'
    X, y = sklearn_datasets.make_regression(
        n_samples=1000,
        n_features=17,
        n_targets=1,
        n_informative=17
    )
    X = pd.DataFrame(
        X,
        columns= [ "col_" +str(i) for i in range(0, 17) ]
    )
    X[coly] = y.reshape(-1, 1)
    y = pd.DataFrame(y.reshape(-1, 1), columns=['y'])

    log('X',X)
    log('y', y)

    return df, colnum, colcat, coly



def test_template(nrows=1000):
    """
        nrows : take first nrows from dataset
    """
    #### Regression PLEASE RANDOM VALUES AS TEST
    ### Fake Regression dataset
    df, colcat, colnum, coly = test_dataset_regress_fake()

    X = df[colcat+ colnum]
    y = df[coly]


    # Split the df into train/test subsets
    X_train_full, X_test, y_train_full, y_test = train_test_split(X, y, test_size=0.05, random_state=2021, )#stratify=y) Regression no classes to stratify to
    X_train, X_valid, y_train, y_valid         = train_test_split(X_train_full, y_train_full, random_state=2021,)# stratify=y_train_full)
    # num_classes = len(set(y_train_full[coly].values.ravel()))
    log("X_train", X_train)


    cols_input_type_1 = []
    n_sample = 100
    def post_process_fun(y):
        return y_norm(y, inverse=True, mode='boxcox')

    def pre_process_fun(y):
        return y_norm(y, inverse=False, mode='boxcox')


    m = {'model_pars': {
        ### LightGBM API model   #######################################
        # Specify the ModelConfig for pytorch_tabular
        'model_class':  ""

        # Type of target prediction, evaluation metrics
        ,'model_pars' : {'input_width': 18,  'y_width': 1 }


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

    'compute_pars': { 'metric_list': ['accuracy_score', 'median_absolute_error']
                    },

    'data_pars': { 'n_sample' : n_sample,

        'download_pars' : None,

        'cols_input_type' : cols_input_type_1,
        ### family of columns for MODEL  #########################################################
        'cols_model_group': [ 'colnum',
                              'colcat_binto_onehot',
                            ]

        ,'cols_model_group_custom' :  { 'colnum' : colnum,
                                        'colcat' : colcat,
                                        'coly' : coly
                                      }

        ###################################################
        ,'train': {'Xtrain': X_train,
                    'ytrain': y_train,
                        'Xtest': X_valid,
                        'ytest': y_valid},
                'eval': {'X': X_valid,
                        'y': y_valid},
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

    ##### Running loop
    ll = [
        'model_bayesian_pyro.py::BayesianRegression',
    ]
    for cfg in ll:

        # Set the ModelConfig
        m['model_pars']['model_class'] = cfg

        log('Setup model..')
        model = Model(model_pars=m['model_pars'], data_pars=m['data_pars'], compute_pars= m['compute_pars'] )

        log('\n\nTraining the model..')
        fit(data_pars=m['data_pars'], compute_pars= m['compute_pars'], out_pars=None)
        log('Training completed!\n\n')

        log('Predict data..')
        ypred, ypred_proba = predict(Xpred=None, data_pars=m['data_pars'], compute_pars=m['compute_pars'])
        log(f'Top 5 y_pred: {np.squeeze(ypred)[:5]}')


        log('Saving model..')
        save(path= "ztmp/data/output/torch_tabular")
        #  os.path.join(root, 'data\\output\\torch_tabular\\model'))

        log('Load model..')
        model, session = load_model(path="ztmp/data/output/torch_tabular")

        log('Model architecture:')
        log(model.model)

        reset()



if __name__ == "__main__":
    # import fire
    # fire.Fire()
    test()


