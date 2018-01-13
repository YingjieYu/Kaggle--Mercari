#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan  9 02:32:35 2018

@author: yuyingjie
"""

import pandas as pd
import numpy as np
import scipy

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer,FeatureHasher
from sklearn.preprocessing import LabelBinarizer,LabelEncoder,OneHotEncoder
from sklearn.decomposition import PCA,TruncatedSVD

from sklearn.linear_model import Ridge,SGDRegressor
from sklearn.svm import LinearSVR
from sklearn.tree import DecisionTreeRegressor 


from scipy.sparse import coo_matrix, hstack,csr_matrix
import gc

import nltk
from nltk.stem.porter import *
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
from sklearn.feature_extraction import stop_words
import re
import string

train = pd.read_csv('train.tsv',sep = '\t')
test = pd.read_csv('test.tsv',sep = '\t')
#train = pd.read_table('../input/train.tsv', engine='c')
#test = pd.read_table('../input/test.tsv', engine='c')
nrow_train = train.shape[0]
df = pd.concat([train,test],0)
y_train = train['price']
log_y= np.log1p(y_train)

submission: pd.DataFrame = test[['test_id']]

del train, test
gc.collect()
#%% 
# dealing with brand_name
df['brand_name'].fillna(value='unknown', inplace=True)
x_brand = df['brand_name']
lb = LabelBinarizer(sparse_output=True)
x_brand_lb = lb.fit_transform(x_brand)

x_brand = x_brand.astype('str')
fh = FeatureHasher(n_features=3000,input_type="string")
x_brand_hash = fh.fit_transform(x_brand)

#%%
# dealing with catgorical_name
df['category_name'].fillna(value='unknown/unknown/unknown', inplace=True)

def split_text(text):
    try: return text.split('/')
    except: return ('No Label','No Label','No Label')
df['general_cat'],df['subcat_1'],df['subcat_2'] = zip(*df['category_name'].apply(lambda x: split_text(x)))

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

svd = TruncatedSVD(n_components=100,n_iter= 3,algorithm='randomized')
x_cat_all= svd.fit_transform(x_cat_all)
#sorted(svd.explained_variance_ratio_)
#sum(svd.explained_variance_ratio_) # 87%

#%%
# dealing with item_condition_id
x_ici = pd.get_dummies(df['item_condition_id'],sparse = True)
x_ici = scipy.sparse.csr_matrix(x_ici.values)

#%%
#dealing with shipping
x_shipping = pd.get_dummies(df['shipping'],sparse = True)
x_shipping = scipy.sparse.csr_matrix(x_shipping.values)

#%%
#dealing with name
count = CountVectorizer(min_df=12)
x_name_count = count.fit_transform(df["name"])

#%%
# dealing with item_description
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
    
df['item_description'].fillna(value='unknown', inplace=True)
desc_tfidf = TfidfVectorizer(min_df = 10, max_features= 100000,tokenizer=tokenize,ngram_range = (1,3))
x_desc = desc_tfidf.fit_transform(df['item_description']) 

#%%
# putting all together
X = scipy.sparse.hstack((x_ici,x_shipping, x_cat_all, x_desc,x_brand_hash,x_name_count)).tocsr()
#save_sparse_csr('X_test',X_test)
X_train = X[:nrow_train]
X_test = X[nrow_train:]

#%% 
# Building the first Ridge Model
rg1 = Ridge(solver='sag', alpha = 3, fit_intercept=True)
rg1.fit(X_train, log_y)
pred_rg1 = rg1.preidict(X_test)

#%% Stacking by hand
from sklearn.ensemble import (RandomForestClassifier, AdaBoostClassifier, 
                              GradientBoostingClassifier, ExtraTreesClassifier)
from sklearn.svm import SVC
from sklearn.cross_validation import KFold

ntrain = X_train.shape[0]
ntest = X_test.shape[0]
SEED = 0 # for reproducibility
NFOLDS = 5 # set folds for out-of-fold prediction
kf = KFold(ntrain, n_folds= NFOLDS, random_state=SEED)



class SklearnHelper(object):
    def __init__(self, reg, seed=0, params=None):
        params['random_state'] = seed
        self.reg = reg(**params)

    def train(self, x_train, y_train):
        self.reg.fit(x_train, y_train)

    def predict(self, x):
        return self.reg.predict(x)
    
    def fit(self,x,y):
        return self.reg.fit(x,y)
    
    def feature_importances(self,x,y):
        print(self.reg.fit(x,y).feature_importances_)
        
# Out-of-Fold Predictions
def get_oof(reg, X_train, log_y, X_test, lightgbm = False):
    oof_train = np.zeros((ntrain,))
    oof_test = np.zeros((ntest,))
    oof_test_skf = np.empty((NFOLDS, ntest))

    for i, (train_index, test_index) in enumerate(kf):
        x_tr = X_train[train_index]
        y_tr = log_y[train_index]
        x_te = X_train[test_index]
        
        if lightgbm == False:
            reg.train(x_tr, y_tr)
            oof_train[test_index] = reg.predict(x_te)
            oof_test_skf[i, :] = reg.predict(X_test)
        else:            
            model = reg.train(lightgbm_params, train_set=d_train, num_boost_round=7500, valid_sets=watchlist, \
    early_stopping_rounds=50, verbose_eval=500) 
            
            oof_train[test_index] = model.predict(x_te)
            oof_test_skf[i, :] = model.predict(X_test)

    oof_test[:] = oof_test_skf.mean(axis=0)
    return oof_train.reshape(-1, 1), oof_test.reshape(-1, 1)

#first layer
#ridge_params = {'solver':'saga', 'alpha':3}
sgd_params = {'penalty':'l2'}
lsvr_params = {'epsilon':1.5}
dt_params = {'max_depth':2,'min_samples_leaf':10}
"""for best fit of lightGBM
num_leaves : This parameter is used to set the number of leaves to be formed in a tree. Theoretically relation between num_leaves and max_depth is num_leaves= 2^(max_depth). However, this is not a good estimate in case of Light GBM since splitting takes place leaf wise rather than depth wise. Hence num_leaves set must be smaller than 2^(max_depth) otherwise it may lead to overfitting. Light GBM does not have a direct relation between num_leaves and max_depth and hence the two must not be linked with each other.
min_data_in_leaf : It is also one of the important parameters in dealing with overfitting. Setting its value smaller may cause overfitting and hence must be set accordingly. Its value should be hundreds to thousands of large datasets.
max_depth: It specifies the maximum depth or level up to which tree can grow.
"""

lightgbm_params = {
        'learning_rate': 0.624,
        'application': 'regression',
        'max_depth': 3,
        'num_leaves': 99,
        'verbosity': 10,
        'metric': 'RMSE',
        'nthread': 4
    }



#ridge = SklearnHelper(reg = Ridge, seed = SEED, params=ridge_params)
sgd = SklearnHelper(reg = SGDRegressor, seed = SEED, params=sgd_params)
lsvr = SklearnHelper(reg = LinearSVR, seed = SEED, params = lsvr_params)
dt = SklearnHelper(reg = DecisionTreeRegressor, seed = SEED, params = dt_params)


train_X, valid_X, train_y, valid_y = train_test_split(X_train, log_y, test_size = 0.16, random_state = 44) 
d_train = lgb.Dataset(train_X, label=train_y, max_bin=8192)
d_valid = lgb.Dataset(valid_X, label=valid_y, max_bin=8192)
watchlist = [d_train, d_valid]
 

#ridge_oof_train, ridge_oof_test = get_oof(ridge, X_train, y_train, X_test)
sgd_oof_train, sgd_oof_test = get_oof(sgd, X_train, log_y, X_test)
lsvr_oof_train, lsvr_oof_test = get_oof(lsvr, X_train, log_y, X_test)
dt_oof_train, dt_oof_test = get_oof(dt, X_train, log_y, X_test)
lightgbm_oof_train, lightgbm_oof_test = get_oof(lgb, X_train, log_y, X_test,lightgbm = True)

#np.savetxt('lightgbm_oof_test.txt',lightgbm_oof_test,delimiter=',')


# second layer
#base_prediction_train = pd.DataFrame({'SGD': sgd_oof_train.ravel(),
#                                      'linearSVR': lsvr_oof_train.ravel(),
#                                      'DecitionTree': dt_oof_train.ravel()})
#base_prediction_train.head()
x_train_2 = np.concatenate((sgd_oof_train,lsvr_oof_train,dt_oof_train,lightgbm_oof_train), axis = 1)
x_test_2 = np.concatenate((sgd_oof_test, lsvr_oof_test, dt_oof_test,lightgbm_oof_test), axis = 1)

# make predictionn using randomforestregressor
#regr2 = RandomForestRegressor(n_jobs=-1,n_estimators=100,max_depth=2,min_samples_leaf=10,verbose=1)
#regr2.fit(x_train_2, y_train)
#pre_regr2 = regr2.predict(x_test_2)     


ridge = Ridge(alpha=.05, copy_X=True, fit_intercept=True, max_iter=100,
      normalize=False, random_state=101, solver='auto', tol=0.001)
ridge.fit(x_train_2, log_y)
pred = ridge.predict(x_test_2)
submission['price'] = np.expm1(pred)
submission.to_csv("submission_1.csv", index=False)
