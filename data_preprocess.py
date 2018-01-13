#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec 29 19:11:37 2017

@author: yuyingjie
"""

#%% comning data
import pandas as pd
import numpy as np
import scipy

from sklearn.linear_model import Ridge, LogisticRegression
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.preprocessing import LabelBinarizer,LabelEncoder,OneHotEncoder
from sklearn.decomposition import PCA,TruncatedSVD
from scipy.sparse import coo_matrix, hstack,csr_matrix
import gc
def save_sparse_csr(filename, array):
    # note that .npz extension is added automatically
    np.savez(filename, data=array.data, indices=array.indices,
             indptr=array.indptr, shape=array.shape)


def load_sparse_csr(filename):
    # here we need to add .npz extension manually
    loader = np.load(filename + '.npz')
    return csr_matrix((loader['data'], loader['indices'], loader['indptr']),
                      shape=loader['shape'])
    
#%%

train = pd.read_csv('train.csv')
test = pd.read_csv('tets.csv')
df = pd.concat([train,test],0)
nrow_train = train.shape[0]
y_train = train['price']

df['brand_name'].isnull().sum()
df['brand_name'].nunique() #5289
df["brand_name"] = df["brand_name"].fillna("unknown")
df["brand_name"].value_counts().index[:10]

#%% brand_name
x_brand = df['brand_name']
lb = LabelBinarizer(sparse_output=True)
x_brand_lb = lb.fit_transform(x_brand)
from sklearn.feature_extraction import FeatureHasher
x_brand = x_brand.astype('str')
fh = FeatureHasher(n_features=1000,input_type="string")
x_brand_hash = fh.fit_transform(x_brand)

#%% catogories
x_cat1 = df['general_cat']
x_cat2 = df['subcat_1']
x_cat3 = df['subcat_2']

x_cat1 = x_cat1.astype('category')
x_cat2 = x_cat2.astype('category')
x_cat3 = x_cat3.astype('category')

lb_cat = LabelBinarizer()
x_cat1 = lb_cat.fit_transform(x_cat1)
x_cat2 = lb_cat.fit_transform(x_cat2)
x_cat3 = lb_cat.fit_transform(x_cat3)

x_cat_all = np.hstack((x_cat1,x_cat2, x_cat3))

x_cat_all = scipy.sparse.csr_matrix(x_cat_all)
#save_sparse_csr('x_cat_all',x_cat_all)

svd = TruncatedSVD(n_components=30,n_iter= 3,algorithm='randomized')
x_cat_all= svd.fit_transform(x_cat_all)
sorted(svd.explained_variance_ratio_)
sum(svd.explained_variance_ratio_)

#%% item_condition_ids
x_ici = pd.get_dummies(df['item_condition_id'],sparse = True)
x_ici = scipy.sparse.csr_matrix(x_ici.values)

#%% shipping
x_shipping = pd.get_dummies(df['shipping'],sparse = True)
x_shipping = scipy.sparse.csr_matrix(x_shipping.values)

#%% name
df['name'].nunique() #1750496
df['name'].isnull().sum() #0

count = CountVectorizer(min_df=10)
x_name_count = count.fit_transform(df["name"])


count_tfidf = TfidfVectorizer(max_features = 21244,ngram_range = (1,2))
x_name_tfidf = count_tfidf.fit_transform(df['name'])

#%% desc
import nltk
from nltk.stem.porter import *
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
from sklearn.feature_extraction import stop_words
import re
import string
stop = set(stopwords.words('english'))
def tokenize(text):
    """
    sent_tokenize(): segment text into sentences
    word_tokenize(): break sentences into words
    """
    try: 
        regex = re.compile('[' +re.escape(string.punctuation) + '0-9\\r\\t\\n]')
        text = regex.sub(" ", text) # remove punctuation
        
        tokens_ = [word_tokenize(s) for s in sent_tokenize(text)]
        tokens = []
        for token_by_sent in tokens_:
            tokens += token_by_sent
        tokens = list(filter(lambda t: t.lower() not in stop, tokens))
        filtered_tokens = [w for w in tokens if re.search('[a-zA-Z]', w)]
        filtered_tokens = [w.lower() for w in filtered_tokens if len(w)>=3]
        
        return filtered_tokens
            
    except TypeError as e: print(text,e)
    
    
desc_tfidf = TfidfVectorizer(min_df = 10, max_features= 50000,tokenizer=tokenize,ngram_range = (1,2))
x_desc = desc_tfidf.fit_transform(df['item_description']) 

#%%
print([x_ici.shape, x_shipping.shape,x_cat_all.shape, x_desc.shape, x_brand_hash.shape,x_name_tfidf.shape])
X = scipy.sparse.hstack((x_ici,x_shipping, x_cat_all, x_desc,x_brand_hash,x_name_tfidf)).tocsr()
save_sparse_csr('X',X)
X_train = X[:len(train)]
save_sparse_csr('X_train',X_train)

del train,test
gc.collect()

#%%
model = Ridge(solver = "lsqr", fit_intercept=False)

model.fit(X_train, y_train)
X_test = X[X_train.shape[0]:]
preds = model.predict(X_train)


import math
def rmsle(y, y_pred):
	assert len(y) == len(y_pred)
	terms_to_sum = [(y_pred[i] - y[i]) ** 2.0 for i,pred in enumerate(y_pred)]
	return (sum(terms_to_sum) * (1.0/len(y))) ** 0.5

rmsle(y_train,preds)    
