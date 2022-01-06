import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.base import BaseEstimator, TransformerMixin, RegressorMixin, clone
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import RobustScaler, StandardScaler,MinMaxScaler
from sklearn.metrics import mean_squared_error
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.decomposition import PCA, KernelPCA
from sklearn.model_selection import cross_val_score, GridSearchCV, KFold
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, ExtraTreesRegressor
from sklearn.svm import SVR, LinearSVR
from sklearn.linear_model import ElasticNet, SGDRegressor, BayesianRidge
from sklearn.kernel_ridge import KernelRidge
from sklearn.utils import resample
from sklearn.feature_selection import SelectKBest,chi2
#print("executing the lengthy part")
train_set = pd.read_csv('precipitation.csv')
## replace > 0 with 1
train_set[(train_set['PRCP'] > 0)] = 1
train_set.drop(['TAVG','DATE','STATION','NAME'],axis=1,inplace = True)
#train_set['PRCP'].value_counts().plot(kind='bar',figsize = (20,12))
#plt.show()

new_set = resample(train_set[train_set['PRCP']==1] , replace = True , n_samples = train_set[train_set['PRCP'] == 0].shape[0] - train_set[train_set['PRCP'] == 1].shape[0] )
#new_set = resample(sample_set , replace = True , n_samples = 4 )
#print(f"shape of new_set is {new_set.shape[0]} shape of test_set is {train_set[train_set['PRCP'] == 0].shape[0]}")


train_set = pd.concat([train_set,new_set],axis = 0 , ignore_index = True)
##print(train_set.isnull().sum())
null_cols = train_set.columns[train_set.isnull().sum() > 0]
for col in null_cols:
    if(train_set[col].isnull().sum() > 200):
        train_set.drop([col] , axis=1,inplace=True)
    else:
        train_set[col].fillna(value = train_set[col].value_counts().sort_values(ascending = False).index[0] , inplace = True)


copy_set = train_set.copy()
target = copy_set['PRCP']
train_set.drop(['PRCP'],axis=1,inplace = True)
scaler = MinMaxScaler()
train_set = scaler.fit_transform(train_set)


copy_set.drop(['PRCP'],axis = 1,inplace =True)

KBest = SelectKBest(chi2,k=4)
KBest.fit(train_set,target)

plothelper = pd.DataFrame({'features':copy_set.columns,'scores':KBest.scores_,'pvalue':KBest.pvalues_})
#print(KBest.scores_.shape[0])
#print(copy_set.columns.shape[0])
plt.bar(plothelper['features'],plothelper['scores'])
plt.show()
