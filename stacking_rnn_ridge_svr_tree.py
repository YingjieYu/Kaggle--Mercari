#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 16 18:02:13 2018

@author: yuyingjie
"""

import pandas as pd
import numpy as np
import scipy
import gc

import math
from sklearn.preprocessing import LabelEncoder, MinMaxScaler, StandardScaler
from sklearn.cross_validation import train_test_split

from subprocess import check_output

from scipy.sparse import coo_matrix, hstack,csr_matrix



#print(check_output(["ls", "../input"]).decode("utf8"))

train = pd.read_csv('train.tsv',sep = '\t')
test = pd.read_csv('test.tsv',sep = '\t')
train = train.drop(train[(train.price <3)].index)
nrow_train = train.shape[0]
df = pd.concat([train,test],0)
y_train = train['price']
log_y= np.log1p(y_train)

submission: pd.DataFrame = test[['test_id']]

del train,test
gc.collect()

def rmsle(y, y_pred):
    assert len(y) == len(y_pred)
    to_sum = [(y_pred[i] - y[i]) ** 2.0 for i,pred in enumerate(y_pred)]
    return (sum(to_sum) * (1.0/len(y))) ** 0.5



#%%
df.brand_name.fillna(value="missing", inplace=True)
df.item_description.fillna(value="missing", inplace=True)

df = df.drop(['test_id','train_id'],axis = 1)

#%%
# get name and description length    
'''
The length of the description, that is the raw number of words used, 
does have some correlation with price. The RNN might find this out on it's own, 
but since a max depth is used to save computations, it does not always know. 
Description length clearly helps the model, name length maybe not so much. 
Does not hurt the models so leaving name length in.
'''

def wordCount(text):
    try:
        if text == 'missing':
            return 0
        else:
            text = text.lower()
            words  = [w for w in text.split(" ")]
            return len(words)
    except:
        return 0

df['len_desc'] = df['item_description'].apply(lambda x: wordCount(x))
df['len_name'] = df['name'].apply(lambda x : wordCount(x))


#%% deal with category_name
df['category_name'].fillna(value='unknown/unknown/unknown', inplace=True)

def split_text(text):
    try: return text.split('/')
    except: return ('No Label','No Label','No Label')
df['general_cat'],df['subcat_1'],df['subcat_2'] = zip(*df['category_name'].apply(lambda x: split_text(x)))

#le = LabelEncoder()
#df.category_name = le.fit_transform(df.category_name)
#%% dealing with  brand_name

#df.brand_name = le.fit_transform(df.brand_name)
#df.head()
#del le
#gc.collect()

'''
The brand name data is sparse, missing over 600,000 values. 
This gets some of those values back by checking their names. 
However, It does not seem to help the models either way at this point. 
An exact name match against all_brand names will find about 3000 of these. 
We can be pretty confident in these. At the other extreme, 
we can search for any matches throughout all words in name. 
This finds over 200,000 but a lot of these are incorrect. 
Can land somewhere in the middle by either keeping cases or trimming out some of the 5000 brand names.

For example, PINK is a brand by victoria secret. If we remove case, then almost all pink items are 
labeled as PINK brand. The other issue is that some of the "brand names" are not brands but really categories 
like "Boots" or "Keys".

Currently, checking every word in name of a case-sensitive match does best. 
This gets around 137,000 finds while avoiding the problems with brands like PINK.
'''
all_brands = set(df['brand_name'].values)
permissing = len(df.loc[df['brand_name'] == 'missing'])



def brandfinder(line):
    brand = line[0]
    name = line[1]
    namesplit = name.split(' ')
    if brand == 'missing':
        for x in namesplit:
            if x in all_brands:
                return name
    if name in all_brands:
        return name
    return brand
df['brand_name']  = df[['brand_name','name']].apply(brandfinder, axis = 1)
found = permissing-len(df.loc[df['brand_name'] == 'missing'])
print(found)



#%% dealing with categorical data for nn
le = LabelEncoder()
df['brand_name'] = le.fit_transform(df['brand_name'])
df['general_cat']= le.fit_transform(df['general_cat'])
df['subcat_1'] = le.fit_transform(df['subcat_1'])
df['subcat_2'] = le.fit_transform(df['subcat_2'])
df['category_name']= le.fit_transform(df['category_name'])

del le
print('process categorical features done')


#%% dealing with text for nn
from keras.preprocessing.text import Tokenizer

raw_text=np.hstack([df.item_description.str.lower(), df.name.str.lower()])
tokenizer = Tokenizer()
tokenizer.fit_on_texts(raw_text)

df['seq_item_description'] = tokenizer.texts_to_sequences(df.item_description.str.lower())
df['seq_name'] = tokenizer.texts_to_sequences(df.name.str.lower())
del tokenizer


# sequence variables analysis
max_seq_name = np.max(df['seq_name'].apply(lambda x: len(x)))
print(max_seq_name) #17

max_seq_item_desc = np.max(df['seq_item_description'].apply(lambda x: len(x)))
print(max_seq_item_desc) #269

#df['seq_item_description'].apply(lambda x: len(x)).hist()
#df['seq_name'].apply(lambda x: len(x)).hist()

MAX_SEQ_NAME = 10
MAX_SEQ_ITEM_DESC = 75
MAX_TEXT = np.max([np.max(df['seq_name'].max()),np.max(df['seq_item_description'].max())])+100
MAX_CATEGORY_NAME = np.max(df.category_name.max())+1
MAX_CAT_1 = np.max(df['general_cat'].max()) +1
MAX_CAT_2 = np.max(df['subcat_1'].max()) +1
MAX_CAT_3 = np.max(df['subcat_2'].max()) +1
MAX_CONDITION = np.max(df['item_condition_id'].max()) +1
MAX_BRAND = np.max(df['brand_name'].max()) +1
MAX_LEN_NAME = np.max(df.len_name.max())+1
MAX_LEN_DESC = np.max(df.len_desc.max()) +1

#%% data definition
from keras.preprocessing.sequence import pad_sequences

def get_keras_data(df):
    X = {
     'name':pad_sequences(df['seq_name'], maxlen= MAX_SEQ_NAME),
     'item_desc': pad_sequences(df['seq_item_description'], maxlen = MAX_SEQ_ITEM_DESC),
     'brand_name': np.array(df['brand_name']),
     'category_name':np.array(df['category_name']),
     'subcat_1': np.array(df.general_cat),
     'subcat_2':np.array(df.subcat_1),
     'subcat_3':np.array(df.subcat_2),
     'item_condition_id':np.array(df['item_condition_id']),
     'shipping':np.array(df[['shipping']]),
     'len_name':np.array(df[['len_name']]),
     'len_desc':np.array(df[['len_desc']])}
    return X

train = df[:nrow_train]
test = df[nrow_train:]
train.shape
test.shape
#df_train,df_valid = train_test_split(train, train_size = 0.99)


X_train = get_keras_data(train)
y_train = np.log1p(train.price.values).reshape(-1,1)

#X_valid = get_keras_data(df_valid)
#y_valid = np.log1p(df_valid.price.values).reshape(-1,1)

X_test = get_keras_data(test)

#%% keras model
from keras.layers import Input, Dropout, Dense, BatchNormalization, Activation, concatenate, GRU, Embedding, Flatten, BatchNormalization
from keras.models import Model
from keras.callbacks import ModelCheckpoint, Callback, EarlyStopping
from keras import backend as K
from keras.optimizers import Adam

def get_callbacks(filepath, patience = 2):
    es = EarlyStopping('val_loss',patience = patience, mode = 'min')
    msave = ModelCheckpoint(filepath, save_best_only = True)
    return [es, msave]

def rmsle_cust(y_true, y_pred):
    first_log = K.log(K.clip(y_pred, K.epsilon(), None) +1.)
    second_log = K.log(K.clip(y_true, K.epsilon(), None) +1.)
    return K.sqrt(K.mean(K.square(first_log - second_log), axis = -1))

def get_model(lr = 0.001, decay = 0.0):
     #Inputs
    name = Input(shape = [X_train['name'].shape[1]], name = 'name')
    item_desc = Input(shape = [X_train['item_desc'].shape[1]], name = 'item_desc')
    brand_name= Input(shape = [1], name = 'brand_name')
    category_name = Input(shape = [1], name = 'category_name')
    item_condition_id = Input(shape = [1], name = 'item_condition_id')
    shipping = Input(shape = [1], name = 'shipping')
    subcat_1 = Input(shape = [1], name = 'subcat_1')
    subcat_2 = Input(shape = [1], name = 'subcat_2')
    subcat_3 = Input(shape = [1], name = 'subcat_3')
    len_name = Input(shape = [1], name = 'len_name')
    len_desc = Input(shape = [1], name = 'len_desc')

       
    # Embedding to adjust output to help model
    emb_name = Embedding(MAX_TEXT, 20)(name)
    emb_item_desc = Embedding(MAX_TEXT,60)(item_desc)
    emb_brand_name = Embedding(MAX_BRAND,10)(brand_name)
    emb_category_name = Embedding(MAX_CATEGORY_NAME,10)(category_name)
    emb_item_condition = Embedding(MAX_CONDITION,5)(item_condition_id)
    emb_len_name = Embedding(MAX_LEN_NAME,5)(len_name)
    emb_len_desc = Embedding(MAX_LEN_DESC,5)(len_desc)
    emb_subcat_1 = Embedding(MAX_CAT_1,10)(subcat_1)
    emb_subcat_2 = Embedding(MAX_CAT_2,10)(subcat_2)
    emb_subcat_3 = Embedding(MAX_CAT_3,10)(subcat_3)

    #rnn( GRU is faster than LSTMs)
    rnn_1 = GRU(16)(emb_item_desc)
    rnn_2 = GRU(8)(emb_name)
      
    # main layer
    main_1 = concatenate([
       Flatten()(emb_brand_name),
       Flatten()(emb_category_name),
       Flatten()(emb_item_condition),
       Flatten()(emb_len_name),
       Flatten()(emb_len_desc),
       Flatten()(emb_subcat_1),
       Flatten()(emb_subcat_2),
       Flatten()(emb_subcat_3),
       rnn_1,
       rnn_2,
       shipping])
    
    main_1 = Dropout(0.1)(Dense(512, kernel_initializer='normal', activation='relu')(main_1))
    main_1 = Dropout(0.1)(Dense(256, kernel_initializer='normal', activation='relu')(main_1))
    main_1 = Dropout(0.1)(Dense(128, kernel_initializer='normal', activation='relu')(main_1))
    main_1 = Dropout(0.1)(Dense(64,kernel_initializer='normal', activation='relu')(main_1))
       
    # output
    output = Dense(1, activation='linear')(main_1)
    
    #model
    model = Model([name,item_desc,brand_name,category_name,item_condition_id,shipping,
                   len_name,len_desc, subcat_1,subcat_2, subcat_3],output)
    optimizer = Adam(lr = lr, decay = decay)
    model.compile(loss='mse',optimizer = optimizer,metrics = ["mae", rmsle_cust])
    
    return model

model = get_model()
model.summary()
del model


#%%
'''
 It takes around 35-40 minutes to run the RNN model. 
 2 epochs with smaller batches tends to do better than more epochs with larger batches. 
 Trimming time off here will be important if adding more models.
 '''
BATCH_SIZE = 512*3
epochs = 2 

# Calculate learning rate decay.
exp_decay = lambda init, fin, steps: (init/fin)**(1/(steps-1)) - 1
steps = int(len(X_train['name']) / BATCH_SIZE) * epochs
lr_init, lr_fin = 0.005, 0.001
lr_decay = exp_decay(lr_init, lr_fin, steps)
 
model= get_model(lr=lr_init, decay=lr_decay)

model.fit(X_train,y_train,epochs = epochs, batch_size = BATCH_SIZE, validation_data=(X_valid,y_valid), verbose=1)

#%%
#Evaluating the model on validation data...

#y_valid_preds_rnn = model.predict(X_valid, batch_size=BATCH_SIZE)
#print(" RMSLE error:", rmsle(y_valid, y_valid_preds_rnn))


pred_rnn_train = model.predict(X_train, batch_size =BATCH_SIZE)
pred_rnn_test = model.predict(X_test, batch_size =BATCH_SIZE)
print('rnn model prediction done')
#%% data preprocessing for other models

# transforming to categorical variables
df['brand_name'] = df['brand_name'].astype(str)
df['category_name'] = df['category_name'].astype(str)
df['item_condition_id'] = df['item_condition_id'].astype(str)
df['general_cat'] = df['general_cat'].astype(str)
df['subcat_1'] = df['subcat_1'].astype(str)
df['subcat_2'] = df['subcat_2'].astype(str)
df['len_name'] = df['len_name'].astype(str)
df['len_desc'] = df['len_desc'].astype(str)

print('transforming to cat variables done')

from sklearn.decomposition import PCA,TruncatedSVD
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer,FeatureHasher

# dealing with item_condition_id
X_ici = pd.get_dummies(df['item_condition_id'],sparse = True)
X_ici = scipy.sparse.csr_matrix(X_ici.values)

#dealing with shipping
X_shipping = pd.get_dummies(df['shipping'],sparse = True)
X_shipping = scipy.sparse.csr_matrix(X_shipping.values)

print('shipping/item_condition_id done')
# cats
POP_CAT2 = df['subcat_2'].value_counts().index[:500]
df.loc[~df['subcat_2'].isin(POP_CAT2),'subcat_2'] = '-1'

cv_cat = CountVectorizer()
X_cat1 = cv_cat.fit_transform(df['general_cat'])
X_cat2 = cv_cat.fit_transform(df['subcat_1'])
X_cat3 = cv_cat.fit_transform(df['subcat_2'])

X_cat_all = scipy.sparse.hstack((X_cat1,X_cat2, X_cat3))

svd = TruncatedSVD(n_components=100,n_iter= 3,algorithm='randomized')
X_cat_all= svd.fit_transform(X_cat_all)
X_cat_all = scipy.sparse.csr_matrix(X_cat_all)
# this takes a lot of time, think abot trim it
print('cats done')

cv = CountVectorizer()
X_len_name = cv.fit_transform(df['len_name']) 
X_len_desc = cv.fit_transform(df['len_name'])
X_brand = cv.fit_transform(df['brand_name'])
print('len_name and len_desc and brand_name done')


#import nltk
#from nltk.stem.porter import *
#from nltk.tokenize import word_tokenize, sent_tokenize
#from nltk.corpus import stopwords
#from sklearn.feature_extraction import stop_words
#import re
#import string
#
#stop = set(stopwords.words('english'))
#def tokenize(text):
#    """
#    sent_tokenize(): segment text into sentences
#    word_tokenize(): break sentences into words
#    """
#    try: 
#        regex = re.compile('[' +re.escape(string.punctuation) + '0-9\\r\\t\\n]')
#        text = regex.sub(" ", text) # remove punctuation
#        
#        tokens_ = [word_tokenize(s) for s in sent_tokenize(text)]
#        tokens = []
#        for token_by_sent in tokens_:
#            tokens += token_by_sent
#        tokens = list(filter(lambda t: t.lower() not in stop, tokens))
#        filtered_tokens = [w for w in tokens if re.search('[a-zA-Z]', w)]
#        filtered_tokens = [w.lower() for w in filtered_tokens if len(w)>=3]
#        
#        return filtered_tokens
#            
#    except TypeError as e: print(text,e)


# this takes a lot of time, try tuning!!
#desc_tfidf = TfidfVectorizer(min_df = 10, max_features= 50000,tokenizer=tokenize,ngram_range = (1,3))
#X_desc = desc_tfidf.fit_transform(df['item_description']) 
#print('tf-idf item_description done')


desc_tfidf = TfidfVectorizer(max_features = 50000, 
                              ngram_range = (1,3),
                              stop_words = "english")
X_desc = desc_tfidf.fit_transform(df["item_description"])
print('tf-idf item_description done')


name_cv = CountVectorizer(min_df=10)
X_name_count = name_cv.fit_transform(df["name"])
print('count vectorize name done')


#%%
X = hstack((X_ici,X_shipping, X_cat_all, X_desc,X_brand,X_name_count,X_len_name, X_len_desc)).tocsr()

X_train = X[:nrow_train]
X_test = X[nrow_train:]

del X
del df
gc.collect()
#%%
from sklearn.linear_model import Ridge,SGDRegressor
from sklearn.svm import LinearSVR
from sklearn.tree import DecisionTreeRegressor 

dtr = DecisionTreeRegressor(max_depth = 2,min_samples_leaf=10)
dtr.fit(X_train, log_y)
pred_ftr_train = dtr.predict(X_train)
pred_ftr_test = dtr.predict(X_test)
pred_ftr_train = pred_ftr_train.reshape(-1,1)
pred_ftr_test = pred_ftr_test.reshape(-1,1)


lsvr = LinearSVR(epsilon = 1.5)
lsvr.fit(X_train, log_y)
pred_lsvr_train = lsvr.predict(X_train)
pred_lsvr_test = lsvr.predict(X_test)

pred_lsvr_train = pred_lsvr_train.reshape(-1,1)
pred_lsvr_test = pred_lsvr_test.reshape(-1,1)

pred_rnn_train = pred_rnn_train.reshape(-1,1)
pred_rnn_test = pred_rnn_test.reshape(-1,1)

X_train_2 = np.concatenate((pred_rnn_train, pred_ftr_train,pred_lsvr_train),axis =1)
X_test_2 = np.concatenate((pred_rnn_test, pred_ftr_test,pred_lsvr_test), axis = 1)

ridge = Ridge(alpha=.05, copy_X=True, fit_intercept=True, max_iter=100,
      normalize=False, random_state=101, solver='auto', tol=0.001)
ridge.fit(X_train_2, log_y)
pred_2 = ridge.predict(X_test_2)
submission['price'] = np.expm1(pred_2)
submission.to_csv("submission_2.csv", index=False)

