#============================= IMPORT LIBRARIES =============================

import pandas as pd
from sklearn import preprocessing
from tkinter.filedialog import askopenfilename


#============================= DATA SELECTION ==============================

dataframe=pd.read_csv("Dataset.csv")
print("----------------------------------------")
print("Input Data ")
print("----------------------------------------")
print()
print(dataframe.head(20))
 
#============================= PREPROCESSING ==============================

#==== checking missing values ====

print("----------------------------------------")
print(" Before Checking missing values ")
print("----------------------------------------")
print()
print(dataframe.isnull().sum())

print("-------------------------------------------------")
print(" After Checking Missing Values ")
print("----------------------------------------------")
print()


# Columns to fill with mean
cols_mean_fill = ['PM2.5', 'PM10', 'NO', 'NO2', 'NOx', 'NH3', 'CO', 'SO2',
                  'O3', 'Benzene', 'Toluene', 'Xylene', 'AQI']

# Fill NaNs in these columns with their respective means
for col in cols_mean_fill:
    dataframe[col].fillna(dataframe[col].mean(), inplace=True)

# Fill NaNs in AQI_Bucket with 0
dataframe['AQI_Bucket'].fillna(0, inplace=True)

# Verify missing values after fill
print(dataframe.isnull().sum())

#==== drop unwanted columns ====


cols=['Date']

dataframe=dataframe.drop(cols, axis = 1)

#==== label encoding ====


print("-----------------------------------------------")
print(" Before label Encoding")
print("-----------------------------------------------")
print()

print(dataframe['AQI_Bucket'].tail(20))


label_encoder = preprocessing.LabelEncoder()

dataframe['AQI_Bucket'] = label_encoder.fit_transform(dataframe['AQI_Bucket'].astype(str))

dataframe['City'] = label_encoder.fit_transform(dataframe['City'].astype(str))


print("---------------------------------------")
print("After label Encoding")
print("--------------------------------------")
print()

print(dataframe['AQI_Bucket'].tail(20))


#========================= DATA SPLITTING ============================



from sklearn.model_selection import train_test_split

x=dataframe.drop(['AQI','AQI_Bucket','City'],axis=1)
y=dataframe['AQI']



X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=2)

print("----------------------------------------")
print(" Data splitting ")
print("----------------------------------------")
print()
print("Total data    :",dataframe.shape[0])
print()
print("Training data :",X_train.shape[0])
print()
print("Testing data :",X_test.shape[0])
print()

#========================= CLASSIFICATION ============================


print("---------------------------------")
print(" Classification ")
print("--------------------------------")
print()

#==== LINEAR REGRESSION ====

from sklearn import linear_model
from sklearn import metrics

#initialize the model

logreg = linear_model.LinearRegression()

#fitting the model
logistic = logreg.fit(X_train,y_train)

#predict the model
y_pred_lr = logistic.predict(X_test)

print("----------------------------------------")
print("Linear Regression ")
print("----------------------------------------")
print()
mae_lr=metrics.mean_absolute_error(y_pred_lr,y_test)/10
print("1) Mean Absulte Error  = ", mae_lr)
print()
import numpy as np
mse_lr=metrics.mean_squared_error(y_pred_lr,y_test)
rmse = np.sqrt(mse_lr)/10

print("2) Root Mean Squared Error = ", rmse)


######

x=dataframe.drop(['AQI','AQI_Bucket'],axis=1)
y=dataframe['AQI_Bucket']



X_train1, X_test1, y_train1, y_test1 = train_test_split(x, y, test_size=0.25, random_state=2)

print("----------------------------------------")
print(" Data splitting ")
print("----------------------------------------")
print()
print("Total data    :",dataframe.shape[0])
print()
print("Training data :",X_train.shape[0])
print()
print("Testing data :",X_test.shape[0])
print()

#========================= CLASSIFICATION ============================


print("---------------------------------")
print(" Classification ")
print("--------------------------------")
print()

#==== RANDOM FOREST CLASSIFIER ====

from sklearn.ensemble import RandomForestClassifier,VotingClassifier
from sklearn import metrics

#initialize the model

rf = RandomForestClassifier()

#fitting the model
rf.fit(X_train1,y_train1)

#predict the model
y_pred_rf = rf.predict(X_train1)

print("----------------------------------------")
print("Random Forest Classifier ")
print("----------------------------------------")
print()

acc_rf = metrics.accuracy_score(y_train1, y_pred_rf)*100

error_rf = 100-acc_rf


print("1) Accuracy = ", acc_rf)
print()
print("2) Error rate = ", error_rf)
print()
print("3) Classification Report")
print()
print(metrics.classification_report(y_train1, y_pred_rf))



#====  HYBRID CLASSIFIER ====

from sklearn.linear_model import PassiveAggressiveClassifier

# Initialize classifiers
pac = PassiveAggressiveClassifier()
rf = RandomForestClassifier()

voting_clf = VotingClassifier(
    estimators=[('pac', pac), ('rf', rf)],
    voting='hard'  
)

# Train
voting_clf.fit(X_train1, y_train1)

# Predict
y_pred = voting_clf.predict(X_train1)


acc_hyb = metrics.accuracy_score(y_train1, y_pred)*100


y_pred1 = voting_clf.predict(X_test1)


acc_hyb1 = metrics.accuracy_score(y_test1, y_pred1)*100


print("----------------------------------------")
print("Hybrid Classifier ")
print("----------------------------------------")
print()

print("1) Accuracy = ", acc_hyb)
print()
print("2) Error rate = ", error_rf)
print()
print("3) Classification Report")
print()
print(metrics.classification_report(y_train1, y_pred_rf))


import pickle

# After training Linear Regression model (logistic)
with open('linear_regression_model.pkl', 'wb') as f:
    pickle.dump(logistic, f)
print("Linear Regression model saved as linear_regression_model.pkl")

# After training Random Forest model (rf)
with open('random_forest_model.pkl', 'wb') as f:
    pickle.dump(rf, f)
print("Random Forest model saved as random_forest_model.pkl")