from petastorm.reader import Reader, make_batch_reader
from petastorm.tf_utils import make_petastorm_dataset
import petastorm

def get_dataset_split_for_model_petastorm(Xtrain, ytrain=None, pars:dict=None):
    """  Split data for moel input/
    Xtrain  ---> Split INTO  tuple PetaStorm Reader
    https://github.com/uber/petastorm/blob/master/petastorm/reader.py#L61-L134
    :param Xtrain:  path
    :param cols_type_received:
    :param cols_ref:
    :return:
    """
    

    dataset_url_train = Xtrain
    batch_size = 128
    num_classes = 10
    epochs = 12
    file_path = 'C:/Users/TusharGoel/Desktop/Upwork/project4/dsa2/'+ dataset_url_train
    file = "file:///" + file_path
    print(file)
    with make_batch_reader(file) as reader:
        train_reader  = make_batch_reader(reader, num_epochs=epochs)
    train_dataset = make_petastorm_dataset(train_reader)

    ### Re-shape  #############################################
    train_dataset = train_dataset.map(lambda x: (tf.reshape(x.image, (28, 28, 1)), tf.reshape(x.digit, [1])))



    ###########################################################
    train_dataset = train_dataset.batch(batch_size, drop_remainder=True)
    return train_dataset


print(get_dataset_split_for_model_petastorm('datasets/parquet/f01.parquet'))

