
#### Check here if latest commit is working :

[Testing code ](https://github.com/arita37/dsa2/blob/main/ztest/run_fast.sh)

Main
![Main, test_fast_linux](https://github.com/arita37/dsa2/workflows/test_fast_linux/badge.svg?branch=main)
![Main, test_full](https://github.com/arita37/dsa2/workflows/test_full/badge.svg?branch=main)


Multi
  ![test_fast_linux](https://github.com/arita37/dsa2/workflows/test_fast_linux/badge.svg?branch=multi)
   ![test_full](https://github.com/arita37/dsa2/workflows/test_full/badge.svg?branch=multi)


Preprocessors Check
![test_preprocess](https://github.com/arita37/dsa2/workflows/test_preprocess/badge.svg?branch=multi)


### Looking for contributors
     Maintain and setup roadmap of this excellent Data Science / ML repo.
     Goal is to unified Data Science and Machine Learning .
     Basic idea is to have one single dictionary/json for
            model, compute, data definition,
     --> easy to define, easy to track, easy to modify.
     

### Install 
     pip install -r zrequirements.txt


### Basic usage 
    python  titanic_classifier.py  preprocess    --nsample 1000
    python  titanic_classifier.py  train         --nsample 2000
    python  titanic_classifier.py  predict



### How to train a new dataset ?
    1) Put your data file   in   data/input/mydata/raw/   
[link](https://github.com/arita37/dsa2/tree/multi/data/input/mydata)
       

    2) Update script        in   data/input/mydata/clean.py
       to load column names, basic profile...

    3) Run  python clean.py profile   and check results


    4) run  python clean.py train_test
        which generates train and test data in :   
           data/input/mydata/train/features.parquet   target.parquet  (y label)        
           data/input/mydata/test/features.parquet    target.parquet  (y label)                
                

    4) Copy Paste titanic_classifier.py  into  mydata_classifier.py
    
    5) Modify the script     mydata_classifier.py
        to match your dataset and the models you want to test.
          
    6) Run 
        python  mydata_classifier.py  train
        python  mydata_classifier.py  predict


        
### Examples

      In example/




### List of preprocessor

        prepro_sampler.pd_autoencoder
        prepro_sampler.pd_col_genetic_transform
        prepro_sampler.pd_colcat_encoder_generic
        prepro_sampler.pd_filter_resample
        prepro_sampler.pd_filter_rows


        prepro.pd_autoencoder
        prepro.pd_col_covariate_shift_adjustment
        prepro.pd_col_genetic_transform
        prepro.pd_colcat_bin
        prepro.pd_colcat_encoder_generic
        prepro.pd_colcat_minhash
        prepro.pd_colcat_to_onehot
        prepro.pd_colcross
        prepro.pd_coldate
        prepro.pd_colnum
        prepro.pd_colnum_bin
        prepro.pd_colnum_binto_onehot
        prepro.pd_colnum_normalize
        prepro.pd_colnum_quantile_norm
        prepro.pd_coltext
        prepro.pd_coltext_clean
        prepro.pd_coltext_universal_google
        prepro.pd_coltext_wordfreq
        prepro.pd_coly
        prepro.pd_filter_resample
        prepro.pd_filter_rows
        prepro.pd_label_clean


        prepro_tseries.pd_ts_autoregressive
        prepro_tseries.pd_ts_basic
        prepro_tseries.pd_ts_date
        prepro_tseries.pd_ts_detrend
        prepro_tseries.pd_ts_generic
        prepro_tseries.pd_ts_groupby
        prepro_tseries.pd_ts_identity
        prepro_tseries.pd_ts_lag
        prepro_tseries.pd_ts_onehot
        prepro_tseries.pd_ts_rolling
        prepro_tseries.pd_ts_template


















