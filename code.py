# --------------
#intitilize the library 
import pandas as pd
import numpy as np
import os
import seaborn as sns
import matplotlib.pyplot as plt
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectFromModel      
from sklearn.feature_selection import RFE
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score,classification_report,confusion_matrix

                                                                                                                                             
#load the data
seg=pd.read_csv(path,encoding='latin1')

#categorical columns
categorical = [var for var in seg.columns if seg[var].dtype=='O']

#numerical columns
numerical = [var for var in seg.columns if seg[var].dtype!='O']

# concat the numerical variable and categorical columns
df = pd.concat([seg[numerical], # numerical variables 
                      pd.get_dummies(seg[categorical])],   # binary categorical variable
                     axis=1)




# --------------
# independent variable
X=df.drop(['customer id','segments'], axis=1)

#dependent variable
y=df['segments']

# split the dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=42, shuffle = True)

# Instantiate the XGBoost calssifier
classifier = xgb.XGBClassifier(random_state=2)
classifier.fit(X_train, y_train)

#Making prediction on test data
y_pred = classifier.predict(X_test)


f1 = f1_score(y_test, y_pred, average='macro')
print('f1_score:\n',f1)

#Finding the classification report
xgb_cr=classification_report(y_test,y_pred)
print('Classification report: \n', xgb_cr)




# --------------
from sklearn.model_selection import GridSearchCV
parameters={'learning_rate':[0.1,0.15,0.2,0.25,0.3],
            'max_depth':range(1,3)}


# code start here
grid_search = GridSearchCV(estimator=classifier, param_grid=parameters,n_jobs=-1, verbose=4)
grid_search.fit(X_train, y_train)
grid_predictions = grid_search.predict(X_test)
grid_f1 = f1_score(y_test, grid_predictions, average='macro')
print(grid_f1)
report = classification_report(y_test, grid_predictions)
print(report)

# code ends here


# --------------
from sklearn.ensemble import RandomForestClassifier


# code starts here
model = RandomForestClassifier(random_state=2)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
f1 = f1_score(y_test, y_pred, average='macro')
print(f1)
report = classification_report(y_test, y_pred)
print(report)
# code ends here


