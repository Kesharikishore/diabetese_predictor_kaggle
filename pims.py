
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
#import seaborn as sns
from sklearn.model_selection import train_test_split
import xgboost
from sklearn.impute import SimpleImputer
from sklearn.model_selection import RandomizedSearchCV
import warnings
from sklearn.model_selection import cross_val_score
import pickle
warnings.filterwarnings("ignore")
df=pd.read_csv(r'C:\\Users\\kesha\\machine learning\\diabetese prediction\\diabetes.csv')
df1=df.copy()
X=df1.iloc[:,:-1]
Y=df1.iloc[:,-1:]
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size = 0.30, random_state=10)
fill_values = SimpleImputer(missing_values=0, strategy="mean")
X_train = fill_values.fit_transform(X_train)
X_test = fill_values.fit_transform(X_test)
params={
 "learning_rate"    : [0.05, 0.10, 0.15, 0.20, 0.25, 0.30 ] ,
 "max_depth"        : [ 3, 4, 5, 6, 8, 10, 12, 15],
 "min_child_weight" : [ 1, 3, 5, 7 ],
 "gamma"            : [ 0.0, 0.1, 0.2 , 0.3, 0.4 ],
 "colsample_bytree" : [ 0.3, 0.4, 0.5 , 0.7 ]
    
}
classifier=xgboost.XGBClassifier()
random_search=RandomizedSearchCV(classifier,param_distributions=params,n_iter=5,scoring='roc_auc',n_jobs=-1,cv=5,verbose=3)
random_search.fit(X_train,y_train)

classifier=xgboost.XGBClassifier(base_score=0.5, booster='gbtree', colsample_bylevel=1,
              colsample_bynode=1, colsample_bytree=0.3, gamma=0.1, gpu_id=-1,
              importance_type='gain', interaction_constraints='',
              learning_rate=0.05, max_delta_step=0, max_depth=3,
              min_child_weight=7, missing=np.nan, monotone_constraints='()',
              n_estimators=100, n_jobs=0, num_parallel_tree=1, random_state=0,
              reg_alpha=0, reg_lambda=1, scale_pos_weight=1, subsample=1,
              tree_method='exact', validate_parameters=1, verbosity=None)

score=cross_val_score(classifier,X,Y,cv=10)
score.mean()
X1=np.array(df1.iloc[:,:-1])
Y1=np.array(df1.iloc[:,-1:])
classifier.fit(X1,Y1)
l1=[[4,148,80,35,0,33.6,0.527,50]]
l2=np.array(l1)
new_model=classifier.predict(l2)
with open ('my_model','wb') as f:
	pickle.dump(classifier,f)