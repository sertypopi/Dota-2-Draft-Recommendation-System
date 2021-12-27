import numpy as np
import pandas as pd
import requests
import json
import urllib
import tensorflow as tf
import tensorflow_datasets
from tensorflow import keras
from gensim.models.callbacks import CallbackAny2Vec
from gensim.models import Word2Vec

def preds(arr):
    
    data=np.reshape(arr,(1,arr.shape[0],arr.shape[1]))
    data=tf.data.Dataset.from_tensor_slices(data)
    data=data.map(lambda x: x[:-1])
    data=data.batch(1,drop_remainder=True)
    return data

def loss3(y_true,y_pred):
    fun=keras.losses.CategoricalCrossentropy()

    
    preds=tf.math.top_k(y_pred,3).indices
    answer=tf.cast(tf.argmax(y_true),tf.int32)
    return tf.where(answer in preds,0,tf.convert_to_tensor(fun(y_true,y_pred),tf.float32))

def predictions(arr):
    model=keras.models.load_model('lstm128_30_customloss.h5',custom_objects={'loss3':loss3})
    embed_df=pd.read_csv('hero_to_embedding.csv',index_col='Heroes')
    instance=[]
    # label=embed_df.loc[arr].index[-1]
    for i in embed_df.loc[arr].index:
        instance.append(embed_df.loc[i][embed_df.columns].values)
    
    
    instance=np.array(instance)
    instance=preds(instance)


    predictions=model.predict(instance)
    ind=np.argpartition(predictions[0],-3)[-3:][::-1]
    
    return ind