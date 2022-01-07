import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.base import BaseEstimator, TransformerMixin, RegressorMixin, clone
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import RobustScaler, StandardScaler,MinMaxScaler
from sklearn.metrics import mean_squared_error,roc_auc_score,precision_score,f1_score,recall_score,confusion_matrix
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.decomposition import PCA, KernelPCA
from sklearn.model_selection import cross_val_score, GridSearchCV, KFold,train_test_split
from sklearn.linear_model import LinearRegression,LogisticRegression
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, ExtraTreesRegressor
from sklearn.svm import SVR, LinearSVR
from sklearn.linear_model import ElasticNet, SGDRegressor, BayesianRidge
from sklearn.kernel_ridge import KernelRidge
from sklearn.utils import resample
from sklearn.feature_selection import SelectKBest,chi2
from sklearn.tree import DecisionTreeClassifier
import joblib
train_set = pd.read_csv('precipitation.csv')


class InitialFixing(BaseEstimator,TransformerMixin):
    def __init__(self):
        pass
    def fit(self, X,y = None):
        return self
    def transform(self,t_set):
        train_set = t_set.copy()
        train_set[(train_set['PRCP'] > 0)] = 1
        train_set.drop(['TAVG','DATE','STATION','NAME'],axis=1,inplace = True)
        new_set = resample(train_set[train_set['PRCP']==1] , replace = True , n_samples = train_set[train_set['PRCP'] == 0].shape[0] - train_set[train_set['PRCP'] == 1].shape[0] )
        train_set = pd.concat([train_set,new_set],axis = 0 , ignore_index = True)
        return train_set

class HandleMissingValues(BaseEstimator,TransformerMixin):
    def __init__(self):
        pass
    def fit(self,X,y=None):
        return self
    def transform(self , t_set):
        train_set = t_set.copy()
        null_cols = train_set.columns[train_set.isnull().sum() > 0]
        for col in null_cols:
            if(train_set[col].isnull().sum() > 200):
                train_set.drop([col] , axis=1,inplace=True)
            else:
                train_set[col].fillna(value = train_set[col].value_counts().sort_values(ascending = False).index[0] , inplace = True)


        return train_set

def params(model_log,X_train,y_train,X_test,y_test,index):
    model_log.fit(X_train,y_train)
    prediction =  model_log.predict(X_test)
    set2 = np.asarray(y_test)
    prediction = np.asarray(prediction)
    metrics = [f1_score(prediction,y_test) , recall_score(prediction,y_test) , roc_auc_score(prediction,y_test) , precision_score(prediction,y_test)]
    metric_name = ['f1_score' , 'recall_score' , 'roc_auc_score' , 'precision_score']
    cf = confusion_matrix(set2,prediction)
    print(cf)
    plt.subplot(2,1,index)
    plt.bar(metric_name,metrics)


pipe = Pipeline([('initial fixing',InitialFixing()),  ('handle missing values' , HandleMissingValues())])
train_set = pipe.fit_transform(train_set)
target = train_set['PRCP']


train_set.drop(['PRCP'] , axis = 1 , inplace = True)
train_set = MinMaxScaler().fit_transform(train_set)


X_train,X_test,y_train,y_test = train_test_split(train_set,target,test_size = 0.3)
y_train = np.asarray(y_train)
y_test = np.asarray(y_test)
params(LogisticRegression(),X_train,y_train,X_test,y_test,1)
params(DecisionTreeClassifier(),X_train,y_train,X_test,y_test,2)
plt.show()
