# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
#%% load data
import pandas as pd
import numpy as np
import scipy
from scipy.sparse import coo_matrix, hstack,csr_matrix
import gc
from sklearn.linear_model import SGDRegressor
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
from sklearn.linear_model import LinearRegression

def save_sparse_csr(filename, array):
    # note that .npz extension is added automatically
    np.savez(filename, data=array.data, indices=array.indices,
             indptr=array.indptr, shape=array.shape)


def load_sparse_csr(filename):
    # here we need to add .npz extension manually
    loader = np.load(filename + '.npz')
    return csr_matrix((loader['data'], loader['indices'], loader['indptr']),
                      shape=loader['shape'])

def rmsle(y, y_pred):
	assert len(y) == len(y_pred)
	terms_to_sum = [(y_pred[i] - y[i]) ** 2.0 for i,pred in enumerate(y_pred)]
	return (sum(terms_to_sum) * (1.0/len(y))) ** 0.5

X_train = load_sparse_csr('X_train')
ntrain = X_train.shape[0]
X = load_sparse_csr('X')
X_test = X[ntrain:]
del X
train = pd.read_csv('train.csv')
y_train = train['price']
del train;gc.collect() 

#%% SGDRegressor
param_grid = [
        {'penalty': ['elasticnet','l1','l2']}]
sgd = SGDRegressor()
grid_search = GridSearchCV(sgd, param_grid, cv = 3)
grid_search.fit(X_train, y_train)
grid_search.best_params_

sgd_elas = SGDRegressor(penalty = 'elasticnet')
sgd_elas.fit_transform(X_train, y_train)
pred_sgd_elas = sgd.predict(X_train)
rmsle(y_train,pred_sgd_elas) # 0.565

sgd_l2 = SGDRegressor(penalty = 'l2')
sgd_l2.fit_transform(X_train,y_train)
pred_sgd_l2 = sgd_l2.predict(X_train)
rmsle(y_train,pred_sgd_l2) # 0.55418


#%%
from sklearn.kernel_ridge import KernelRidge
kr = KernelRidge()
kr.fit(X_train, y_train)
pred_kr = kr.predict(X_train)
rmsle(y_train,pred_kr)

#%%
from sklearn.svm import LinearSVR
from sklearn.preprocessing import MaxAbsScaler
svm_reg = Pipeline((
            ("scaler", MaxAbsScaler()),
            ("svm", LinearSVR(epsilon=1.5)),
        ))
svm_reg.fit(X_train, y_train)
pred_svm1 = svm_reg.predict(X_train)
rmsle(y_train,pred_svm1) #0.6039

# kernel svm
#from sklearn.svm import SVR
#svm_poly_reg = SVR(kernel="poly", degree=2, C=100, epsilon=0.1)
#svm_poly_reg.fit(X_train, y_train)

#%% simple tree
from sklearn.tree import DecisionTreeRegressor 
tree_reg = DecisionTreeRegressor(max_depth=2,min_samples_leaf=10)
tree_reg.fit(X_train, y_train)
tree_pre = tree_reg.predict(X_train)
rmsle(y_train,tree_pre) #0.7136


#%% bagging
from sklearn.ensemble import BaggingRegressor
from sklearn.tree import ExtraTreeRegressor
bag_reg = BaggingRegressor(
            ExtraTreeRegressor(max_depth=2,min_samples_leaf=10), 
            n_estimators=10,
            bootstrap=True, n_jobs=-1,bootstrap_features = True
        )
bag_reg.fit(X_train,y_train)
pre_bag = bag_reg.predict(X_train)
rmsle(y_train,pre_bag) #0.7080

#%%
from sklearn.ensemble import RandomForestRegressor
regr = RandomForestRegressor(max_depth=2,min_samples_leaf=10)
regr.fit(X_train,y_train)
pred_regr = regr.predict(X_train)
rmsle(y_train,pred_regr) #0.7136

#%%

#If your AdaBoost ensemble is overfitting the training set,
# you can try reducing the number of estimators or more strongly regularizing the base estimator.
from sklearn.ensemble import AdaBoostRegressor
ada = AdaBoostRegressor(base_estimator = DecisionTreeRegressor(max_depth=2,min_samples_leaf=10))
ada.fit(X_train,y_train)
pre_ada = ada.predict(X_train)
rmsle(y_train,pre_ada) #0.7327

param_grid_ada = {'learning_rate': [0.1,0.01,0.005],
         'n_estimators':[50,100,200,500]}
grid_search_ada = RandomizedSearchCV(ada, param_distributions = param_grid_ada, n_iter = 10)
grid_search_ada.fit(X_train, y_train)


#%%
import xgboost as xgb
gbm = xgb.XGBClassifier(
    #learning_rate = 0.02,
 n_estimators= 2000,
 max_depth= 4,
 min_child_weight= 2,
 #gamma=1,
 gamma=0.9,                        
 subsample=0.8,
 colsample_bytree=0.8,
 objective= 'binary:logistic',
 nthread= -1,
 scale_pos_weight=1)

gbm.fit(X_train, y_train)



#%% stacking using mlxtend
from mlxtend.regressor import StackingRegressor


lr_stack = LinearRegression()
svr_lr_stack = LinearSVR(epsilon=1.5)
sgd_elas_stack = SGDRegressor(penalty='elasticnet')
tree_reg_stack = DecisionTreeRegressor(max_depth=2,min_samples_leaf=10)

stregr = StackingRegressor(regressors=[lr_stack, svr_lr_stack,sgd_elas_stack], 
                           meta_regressor=tree_reg_stack,verbose=1)
stregr.fit(X_train, y_train)

#from sklearn.model_selection import GridSearchCV
#from sklearn.linear_model import Lasso
#
## Initializing models
#
#lr = LinearRegression()
#svr_lin = SVR(kernel='linear')
#ridge = Ridge(random_state=1)
#lasso = Lasso(random_state=1)
#svr_rbf = SVR(kernel='rbf')
#regressors = [svr_lin, lr, ridge, lasso]
#stregr = StackingRegressor(regressors=regressors, 
#                           meta_regressor=svr_rbf)
#
#params = {'lasso__alpha': [0.1, 1.0, 10.0],
#          'ridge__alpha': [0.1, 1.0, 10.0],
#          'svr__C': [0.1, 1.0, 10.0],
#          'meta-svr__C': [0.1, 1.0, 10.0, 100.0],
#          'meta-svr__gamma': [0.1, 1.0, 10.0]}
#
#grid = GridSearchCV(estimator=stregr, 
#                    param_grid=params, 
#                    cv=5,
#                    refit=True)
#grid.fit(X, y)
#
#for params, mean_score, scores in grid.grid_scores_:
#        print("%0.3f +/- %0.2f %r"
#              % (mean_score, scores.std() / 2.0, params))

#%% stacking by hand
from sklearn.ensemble import (RandomForestClassifier, AdaBoostClassifier, 
                              GradientBoostingClassifier, ExtraTreesClassifier)
from sklearn.svm import SVC
from sklearn.cross_validation import KFold

ntrain = train.shape[0]
ntest = test.shape[0]
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
def get_oof(reg, X_train, y_train, X_test):
    oof_train = np.zeros((ntrain,))
    oof_test = np.zeros((ntest,))
    oof_test_skf = np.empty((NFOLDS, ntest))

    for i, (train_index, test_index) in enumerate(kf):
        x_tr = X_train[train_index]
        y_tr = y_train[train_index]
        x_te = X_train[test_index]

        reg.train(x_tr, y_tr)

        oof_train[test_index] = reg.predict(x_te)
        oof_test_skf[i, :] = reg.predict(X_test)

    oof_test[:] = oof_test_skf.mean(axis=0)
    return oof_train.reshape(-1, 1), oof_test.reshape(-1, 1)

#first layer
sgd_params = {'penalty':'elasticnet'}
lsvr_params = {'epsilon':1.5}
dt_params = {'max_depth':2,'min_samples_leaf':10}

sgd = SklearnHelper(reg = SGDRegressor, seed = SEED, params=sgd_params)
lsvr = SklearnHelper(reg = LinearSVR, seed = SEED, params = lsvr_params)
dt = SklearnHelper(reg = DecisionTreeRegressor, seed = SEED, params = dt_params)

sgd_oof_train, sgd_oof_test = get_oof(sgd, X_train, y_train, X_test)
lsvr_oof_train, lsvr_oof_test = get_oof(lsvr, X_train, y_train, X_test)
dt_oof_train, dt_oof_test = get_oof(dt, X_train, y_train, X_test)

# second layer
#base_prediction_train = pd.DataFrame({'SGD': sgd_oof_train.ravel(),
#                                      'linearSVR': lsvr_oof_train.ravel(),
#                                      'DecitionTree': dt_oof_train.ravel()})
#base_prediction_train.head()
x_train_2 = np.concatenate((sgd_oof_train,lsvr_oof_train,dt_oof_train), axis = 1)
x_test_2 = np.concatenate((sgd_oof_test, lsvr_oof_test, dt_oof_test), axis = 1)

# make predictionn using randomforestregressor
regr2 = RandomForestRegressor(n_jobs=-1,n_estimators=100,max_depth=2,min_samples_leaf=10,verbose=1)
regr2.fit(x_train_2, y_train)
pre_regr2 = regr2.predict(x_test_2)

