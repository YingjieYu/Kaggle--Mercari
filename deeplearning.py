#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan  3 21:32:52 2018

@author: yuyingjie
"""
import tempfile
import tensorflow as tf
import pandas as pd
import numpy as np
import scipy
import gc
from scipy.sparse import coo_matrix, hstack,csr_matrix


def rmsle(y, y_pred):
	assert len(y) == len(y_pred)
	terms_to_sum = [(y_pred[i] - y[i]) ** 2.0 for i,pred in enumerate(y_pred)]
	return (sum(terms_to_sum) * (1.0/len(y))) ** 0.5

df_train = pd.read_csv('train.csv')
df_test = pd.read_csv('tets.csv')
#%% Converting Data into Tensors
LABEL_COLUMN = 'price'
CATEGORICAL_COLUMNS = ['item_condition_id','brand_name','shipping','general_cat','subcat_1','subcat_2']
CONTINUOUS_COLUMNS = ['description_len']
TFIDF = ['name','item_description']

def convert_sparse_matrix_to_sparse_tensor(X):
    coo = X.tocoo()
    indices = np.mat([coo.row, coo.col]).transpose()
    return tf.SparseTensor(indices, coo.data, coo.shape)

tfidf_item_desc = convert_sparse_matrix_to_sparse_tensor(x_desc)
tfidf_name = convert_sparse_matrix_to_sparse_tensor(x_name_tfidf)



def input_fn(df):
  # Creates a dictionary mapping from each continuous feature column name (k) to
  # the values of that column stored in a constant Tensor.
  continuous_cols = {k: tf.constant(df[k].values)
                     for k in CONTINUOUS_COLUMNS}
  # Creates a dictionary mapping from each categorical feature column name (k)
  # to the values of that column stored in a tf.SparseTensor.
  categorical_cols = {k: tf.SparseTensor(
      indices=[[i, 0] for i in range(df[k].size)],
      values=df[k].values,
      dense_shape=[df[k].size, 1])
                      for k in CATEGORICAL_COLUMNS}
  tfidf_cols = {'tfidf_item_desc': tfidf_item_desc,
                'tfidf_name': tfidf_name}
  
  # Merges the two dictionaries into one.
  feature_cols = {**continuous_cols,  **categorical_cols, **tfidf_cols}
  # Converts the label column into a constant Tensor.
  label = tf.constant(df[LABEL_COLUMN].values)
  # Returns the feature columns and the label.
  return feature_cols, label

def train_input_fn():
  return input_fn(df_train)

def eval_input_fn():
  return input_fn(df_test)

#%%Engineering Features
for f in CATEGORICAL_COLUMNS:
    df_train[f] = df_train[f].astype(str)   

df_train['shipping'] = df_train['shipping'].map({'1':'seller','0':'buyer'})
shipping = tf.contrib.layers.sparse_column_with_keys(
        column_name = "shipping", keys = ['seller','buyer'])


df_train['item_condition_id'] = df_train['item_condition_id'].map({'1':'very_bad','2':'bad','3':'fair','4':'good','5':'very_good'})
item_condition_id = tf.contrib.layers.sparse_column_with_keys(
        column_name = "item_condition_id", keys = ['very_bad','bad','fair','good','very_good'])


general_cat = tf.contrib.layers.sparse_column_with_keys(
        column_name = "general_cat", keys = ['Men', 'Electronics', 'Women', 'Home', 'Sports & Outdoors',
       'Vintage & Collectibles', 'Beauty', 'Other', 'Kids', 'No Label', 'Handmade'])

df_train["brand_name"] = df_train["brand_name"].fillna("unknown")
brand_name =  tf.contrib.layers.sparse_column_with_hash_bucket("brand_name", hash_bucket_size=1000)
subcat_1 =  tf.contrib.layers.sparse_column_with_hash_bucket("subcat_1", hash_bucket_size=100)
subcat_2 =  tf.contrib.layers.sparse_column_with_hash_bucket("subcat_2", hash_bucket_size=500)

description_len= tf.contrib.layers.real_valued_column("description_len")
desc_len_buckets = tf.contrib.layers.bucketized_column(description_len, boundaries=[20, 40, 60, 80,100,120])

#cat1_x_cat2_x_cat3 = tf.contrib.layers.crossed_column(
#  [general_cat, subcat_1, subcat_2], hash_bucket_size=int(10000))

#%%
#model_dir = tempfile.mkdtemp()
#m = tf.contrib.learn.LinearRegressor(feature_columns=[
#  shipping, item_condition_id,general_cat, brand_name, subcat_1, subcat_2],
#  model_dir=model_dir)
#
#m.fit(input_fn=train_input_fn,steps = 4000)
#pred = m.predict_scores(input_fn=train_input_fn)
#pred_list = list(pred)

#rmsle_m1 = 0
#for pre,i in zip(pred_list, range(len(df_train))):
#    rmsle_m1 += tf.sqrt(tf.reduce_mean(tf.squared_difference(pre, df_train['price'][i])))  
#    
#Why TF decides to stop the training anyway after? 
#This has to do with the fact that you have set num_epochs=1000 and the default batch_size of numpy_input_fn is 128 (see https://github.com/tensorflow/tensorflow/blob/master/tensorflow/contrib/learn/python/learn/learn_io/numpy_io.py). num_epochs=1000 means that fit method will go through the data at most 1000 times (or 1000 steps, whichever occurs first). That's why fit runs for ceiling(1000 * 6 /128)=47 steps. Setting batch_size to 6 (the size of your training dataset) or num_epochs=None will give you more reasonable results (I suggest setting batch_size to at most 6 since using your training samples cyclically more than once in a single step might not make much sense)
  

m2 =  tf.contrib.learn.LinearRegressor(feature_columns=[
  shipping, item_condition_id, general_cat, brand_name, subcat_1, subcat_2],
  model_dir='/Users/yuyingjie/Documents/GitHub/Kaggle-Mercari',
  optimizer=tf.train.FtrlOptimizer(
    learning_rate=0.1,
    l1_regularization_strength=1.0,
    l2_regularization_strength=1.0))

m2.fit(input_fn=train_input_fn,steps = 4000)
