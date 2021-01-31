
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
     Goal is to unified Data Science and Machine Learning :
      REDUCE BOILERPLATE CODE for daily task.
      

### Install 
     pip install -r zrequirements.txt


### Basic usage 
    python  titanic_classifier.py  preprocess    --nsample 1000
    python  titanic_classifier.py  train         --nsample 2000
    python  titanic_classifier.py  predict



### How to train a new dataset ?
    1) Put your data file   in   data/input/mydata/raw/   
      https://github.com/arita37/dsa2/tree/multi/data/input/mydata
       

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

        

### Dashboard for hyper-params optimization

![image](https://user-images.githubusercontent.com/18707623/106399211-a1d72d80-645a-11eb-8288-a6e3c9dc66ad.png)


![image](https://user-images.githubusercontent.com/18707623/106399223-b287a380-645a-11eb-92a2-475454c83a85.png)



```




