"""
Activity 1
Rachel Nana
September 2018
"""

import os;
os.chdir("C:/Users/rfowl/Documents/Tiber/Data Science Learning Group/Activity 1");

# import packages
import pandas as pd
import sklearn
import numpy as np


# STEP 3
# import telco data 
telco = pd.read_csv('WA_Fn-UseC_-Telco-Customer-Churn.csv');

# data shape
print("Dataset Length: ", len(telco));
print("Dataset Shape: ", telco.shape);

# column names
print("Dataset Columns: ",list(telco.columns.values));
# data types
print("Column Types: \n",telco.dtypes);

# view first 3 rows
print(telco[:3])

# view specific columns
print(telco['TotalCharges'])
print(telco['StreamingTV'])

# view specific row
print(telco.iloc[[3]])

# view frequency counts
print(telco['gender'].value_counts())
print(telco['SeniorCitizen'].value_counts())
print(telco['Partner'].value_counts())
print(telco['Dependents'].value_counts())
print(telco['tenure'].value_counts(sort=True))
print(telco['PhoneService'].value_counts())
print(telco['MultipleLines'].value_counts())
print(telco['InternetService'].value_counts())
print(telco['OnlineSecurity'].value_counts())
print(telco['OnlineBackup'].value_counts())
print(telco['DeviceProtection'].value_counts())
print(telco['TechSupport'].value_counts())
print(telco['StreamingTV'].value_counts())
print(telco['StreamingMovies'].value_counts())
print(telco['Contract'].value_counts())
print(telco['PaperlessBilling'].value_counts())
print(telco['PaymentMethod'].value_counts())
print(telco['MonthlyCharges'].value_counts())
print(telco['TotalCharges'].value_counts())
print(telco['Churn'].value_counts())

#check for nulls
print("tenure nulls: ", telco['tenure'].isnull().sum())
print("MonthlyCharges nulls: ", telco['MonthlyCharges'].isnull().sum())
print("TotalCharges nulls: ", telco['TotalCharges'].isnull().sum())

# select rows where PaymentMethod = 'Mailed check'
print(telco.loc[telco['PaymentMethod'] == 'Mailed check'])


# STEP 4
# data prep
# produce indicators
Male_ind = telco['gender'].eq('Male').mul(1)
Partner_ind = telco['Partner'].eq('Yes').mul(1)
Dependents_ind = telco['Dependents'].eq('Yes').mul(1)
PhoneService_ind = telco['PhoneService'].eq('Yes').mul(1)
MultipleLines_ind = telco['MultipleLines'].eq('Yes').mul(1)
FiberOptic_ind = telco.rename(columns={'InternetService':'FiberOptic'})['FiberOptic'].eq('Fiber optic').mul(1)
DSL_ind = telco.rename(columns={'InternetService':'DSL'})['DSL'].eq('DSL').mul(1)
OnlineSecurity_ind = telco['OnlineSecurity'].eq('Yes').mul(1)
OnlineBackup_ind = telco['OnlineBackup'].eq('Yes').mul(1)
DeviceProtection_ind = telco['DeviceProtection'].eq('Yes').mul(1)
TechSupport_ind = telco['TechSupport'].eq('Yes').mul(1)
StreamingTV_ind = telco['StreamingTV'].eq('Yes').mul(1)
StreamingMovies_ind = telco['StreamingMovies'].eq('Yes').mul(1)
Contract_dummies=pd.get_dummies(telco['Contract'], drop_first=True)
PaperlessBilling_ind = telco['PaperlessBilling'].eq('Yes').mul(1)
PaymentMethod_dummies=pd.get_dummies(telco['PaymentMethod'], drop_first=True)
Churn_ind = telco['Churn'].eq('Yes').mul(1)

# features
X = pd.concat([Male_ind, telco['SeniorCitizen'], Partner_ind, Dependents_ind, telco['tenure'], PhoneService_ind, MultipleLines_ind, FiberOptic_ind, DSL_ind, OnlineSecurity_ind, OnlineBackup_ind, DeviceProtection_ind, TechSupport_ind, StreamingTV_ind, StreamingMovies_ind, Contract_dummies, PaperlessBilling_ind, PaymentMethod_dummies, telco['MonthlyCharges'], telco['TotalCharges']], axis=1)
print(X[:5])
print(X.shape)

# encoding
from sklearn import preprocessing
for column in X.columns:
    if X[column].dtype == type(object):
        le = preprocessing.LabelEncoder()
        X[column] = le.fit_transform(X[column])

# dependent variable
y = Churn_ind
print(y[:5])
print(y.shape)

# split the data
from sklearn.model_selection import train_test_split
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.20, random_state=42)


# STEP 5
# decision tree
from sklearn import tree
clf = tree.DecisionTreeClassifier()
clf = clf.fit(X_tr, y_tr)



# STEP 6
# visualize
from sklearn.externals.six import StringIO  
from IPython.display import Image  
from sklearn.tree import export_graphviz
import pydotplus
dot_data = StringIO()
export_graphviz(clf, out_file=dot_data,  
                filled=True, rounded=True,
                special_characters=True)
graph = pydotplus.graph_from_dot_data(dot_data.getvalue())  
Image(graph.create_png())
# write to file
graph.write_pdf("decision_tree.pdf")


# STEP 7
# 10-fold cross-validation
N = round(len(X_tr)/10)

# 10 test sets
X_test1 = X_tr.iloc[0:N]
X_test2 = X_tr.iloc[N:2*N]
X_test3 = X_tr.iloc[2*N:3*N]
X_test4 = X_tr.iloc[3*N:4*N]
X_test5 = X_tr.iloc[4*N:5*N]
X_test6 = X_tr.iloc[5*N:6*N]
X_test7 = X_tr.iloc[6*N:7*N]
X_test8 = X_tr.iloc[7*N:8*N]
X_test9 = X_tr.iloc[8*N:9*N]
X_test10 = X_tr.iloc[9*N:len(X_tr)]
y_test1 = y_tr.iloc[0:N]
y_test2 = y_tr.iloc[N:2*N]
y_test3 = y_tr.iloc[2*N:3*N]
y_test4 = y_tr.iloc[3*N:4*N]
y_test5 = y_tr.iloc[4*N:5*N]
y_test6 = y_tr.iloc[5*N:6*N]
y_test7 = y_tr.iloc[6*N:7*N]
y_test8 = y_tr.iloc[7*N:8*N]
y_test9 = y_tr.iloc[8*N:9*N]
y_test10 = y_tr.iloc[9*N:len(X_tr)]

# 10 train sets
X_train1 = X_tr.iloc[N+1:len(X_tr)]
X_train2 = X_tr.iloc[0:N].append(X_tr.iloc[2*N:len(X_tr)])
X_train3 = X_tr.iloc[0:2*N].append(X_tr.iloc[3*N:len(X_tr)])
X_train4 = X_tr.iloc[0:3*N].append(X_tr.iloc[4*N:len(X_tr)])
X_train5 = X_tr.iloc[0:4*N].append(X_tr.iloc[5*N:len(X_tr)])
X_train6 = X_tr.iloc[0:5*N].append(X_tr.iloc[6*N:len(X_tr)])
X_train7 = X_tr.iloc[0:6*N].append(X_tr.iloc[7*N:len(X_tr)])
X_train8 = X_tr.iloc[0:7*N].append(X_tr.iloc[8*N:len(X_tr)])
X_train9 = X_tr.iloc[0:8*N].append(X_tr.iloc[9*N:len(X_tr)])
X_train10 = X_tr.iloc[0:9*N]
y_train1 = y_tr.iloc[N+1:len(y_tr)]
y_train2 = y_tr.iloc[0:N].append(y_tr.iloc[2*N:len(y_tr)])
y_train3 = y_tr.iloc[0:2*N].append(y_tr.iloc[3*N:len(y_tr)])
y_train4 = y_tr.iloc[0:3*N].append(y_tr.iloc[4*N:len(y_tr)])
y_train5 = y_tr.iloc[0:4*N].append(y_tr.iloc[5*N:len(y_tr)])
y_train6 = y_tr.iloc[0:5*N].append(y_tr.iloc[6*N:len(y_tr)])
y_train7 = y_tr.iloc[0:6*N].append(y_tr.iloc[7*N:len(y_tr)])
y_train8 = y_tr.iloc[0:7*N].append(y_tr.iloc[8*N:len(y_tr)])
y_train9 = y_tr.iloc[0:8*N].append(y_tr.iloc[9*N:len(y_tr)])
y_train10 = y_tr.iloc[0:9*N]

# train 10 classifiers
clf1 = clf.fit(X_train1, y_train1)
clf2 = clf.fit(X_train2, y_train2)
clf3 = clf.fit(X_train3, y_train3)
clf4 = clf.fit(X_train4, y_train4)
clf5 = clf.fit(X_train5, y_train5)
clf6 = clf.fit(X_train6, y_train6)
clf7 = clf.fit(X_train7, y_train7)
clf8 = clf.fit(X_train8, y_train8)
clf9 = clf.fit(X_train9, y_train9)
clf10 = clf.fit(X_train10, y_train10)

# make predictions
predicted1=clf1.predict(X_test1)
predicted2=clf2.predict(X_test2)
predicted3=clf3.predict(X_test3)
predicted4=clf4.predict(X_test4)
predicted5=clf5.predict(X_test5)
predicted6=clf6.predict(X_test6)
predicted7=clf7.predict(X_test7)
predicted8=clf8.predict(X_test8)
predicted9=clf9.predict(X_test9)
predicted10=clf10.predict(X_test10)

# accuracy scores
from sklearn.metrics import accuracy_score
ac1=accuracy_score(predicted1,y_test1)
ac2=accuracy_score(predicted2,y_test2)
ac3=accuracy_score(predicted3,y_test3)
ac4=accuracy_score(predicted4,y_test4)
ac5=accuracy_score(predicted5,y_test5)
ac6=accuracy_score(predicted6,y_test6)
ac7=accuracy_score(predicted7,y_test7)
ac8=accuracy_score(predicted8,y_test8)
ac9=accuracy_score(predicted9,y_test9)
ac10=accuracy_score(predicted10,y_test10)

# confusion matrices
from sklearn.metrics import confusion_matrix
cm1=confusion_matrix(y_test1, predicted1)
cm2=confusion_matrix(y_test2, predicted2)
cm3=confusion_matrix(y_test3, predicted3)
cm4=confusion_matrix(y_test4, predicted4)
cm5=confusion_matrix(y_test5, predicted5)
cm6=confusion_matrix(y_test6, predicted6)
cm7=confusion_matrix(y_test7, predicted7)
cm8=confusion_matrix(y_test8, predicted8)
cm9=confusion_matrix(y_test9, predicted9)
cm10=confusion_matrix(y_test10, predicted10)

# true positive / false positive rates
# true positives
TP1 = cm1[0][0]
TP2 = cm2[0][0]
TP3 = cm3[0][0]
TP4 = cm4[0][0]
TP5 = cm5[0][0]
TP6 = cm6[0][0]
TP7 = cm7[0][0]
TP8 = cm8[0][0]
TP9 = cm9[0][0]
TP10 = cm10[0][0]
# false positives
FP1 = cm1[0][1]
FP2 = cm2[0][1]
FP3 = cm3[0][1]
FP4 = cm4[0][1]
FP5 = cm5[0][1]
FP6 = cm6[0][1]
FP7 = cm7[0][1]
FP8 = cm8[0][1]
FP9 = cm9[0][1]
FP10 = cm10[0][1]
# false negatives
FN1 = cm1[1][0]
FN2 = cm2[1][0]
FN3 = cm3[1][0]
FN4 = cm4[1][0]
FN5 = cm5[1][0]
FN6 = cm6[1][0]
FN7 = cm7[1][0]
FN8 = cm8[1][0]
FN9 = cm9[1][0]
FN10 = cm10[1][0]
# true negatives
TN1 = cm1[1][1]
TN2 = cm2[1][1]
TN3 = cm3[1][1]
TN4 = cm4[1][1]
TN5 = cm5[1][1]
TN6 = cm6[1][1]
TN7 = cm7[1][1]
TN8 = cm8[1][1]
TN9 = cm9[1][1]
TN10 = cm10[1][1]
# true positive rates
TPR1 = TP1/(TP1+FP1)
TPR2 = TP2/(TP2+FP2)
TPR3 = TP3/(TP3+FP3)
TPR4 = TP4/(TP4+FP4)
TPR5 = TP5/(TP5+FP5)
TPR6 = TP6/(TP6+FP6)
TPR7 = TP7/(TP7+FP7)
TPR8 = TP8/(TP8+FP8)
TPR9 = TP9/(TP9+FP9)
TPR10 = TP10/(TP10+FP10)
# false positive rates
FPR1 = FP1/(FP1+TN1)
FPR2 = FP2/(FP2+TN2)
FPR3 = FP3/(FP3+TN3)
FPR4 = FP4/(FP4+TN4)
FPR5 = FP5/(FP5+TN5)
FPR6 = FP6/(FP6+TN6)
FPR7 = FP7/(FP7+TN7)
FPR8 = FP8/(FP8+TN8)
FPR9 = FP9/(FP9+TN9)
FPR10 = FP10/(FP10+TN10)

# print true positive and false positive rates
print('ITERATION 1')
print('TRUE POSITVE RATE: ',TPR1)
print('FALSE POSITIVE RATE: ',FPR1)
print('ITERATION 2')
print('TRUE POSITVE RATE: ',TPR2)
print('FALSE POSITIVE RATE: ',FPR2)
print('ITERATION 3')
print('TRUE POSITVE RATE: ',TPR3)
print('FALSE POSITIVE RATE: ',FPR3)
print('ITERATION 4')
print('TRUE POSITVE RATE: ',TPR4)
print('FALSE POSITIVE RATE: ',FPR4)
print('ITERATION 5')
print('TRUE POSITVE RATE: ',TPR5)
print('FALSE POSITIVE RATE: ',FPR5)
print('ITERATION 6')
print('TRUE POSITVE RATE: ',TPR6)
print('FALSE POSITIVE RATE: ',FPR6)
print('ITERATION 7')
print('TRUE POSITVE RATE: ',TPR7)
print('FALSE POSITIVE RATE: ',FPR7)
print('ITERATION 8')
print('TRUE POSITVE RATE: ',TPR8)
print('FALSE POSITIVE RATE: ',FPR8)
print('ITERATION 9')
print('TRUE POSITVE RATE: ',TPR9)
print('FALSE POSITIVE RATE: ',FPR9)
print('ITERATION 10')
print('TRUE POSITVE RATE: ',TPR10)
print('FALSE POSITIVE RATE: ',FPR10)

# combined
FN_sum = FN1 + FN2 + FN3 + FN4 + FN5 + FN6 + FN7 + FN8 + FN9 + FN10
FP_sum = FP1 + FP2 + FP3 + FP4 + FP5 + FP6 + FP7 + FP8 + FP9 + FP10
TN_sum = TN1 + TN2 + TN3 + TN4 + TN5 + TN6 + TN7 + TN8 + TN9 + TN10
TP_sum = TP1 + TP2 + TP3 + TP4 + TP5 + TP6 + TP7 + TP8 + TP9 + TP10

TPR_avg = TP_sum/(TP_sum+FP_sum)
FPR_avg = FP_sum/(FP_sum+TN_sum)

print('COMBINED')
print('TRUE POSITVE RATE: ',TPR_avg)
print('FALSE POSITIVE RATE: ',FPR_avg)


# STEP 8
from sklearn.metrics import roc_curve, roc_auc_score
from matplotlib import pyplot as plt

# predictions
predictions = clf.predict_proba(X_te)
print(roc_auc_score(y_te, predictions[:,1]))

# ROC curve
fpr, tpr, _ = roc_curve(y_te, predictions[:,1])
plt.clf()
plt.plot(fpr, tpr)
plt.xlabel('FPR')
plt.ylabel('TPR')
plt.title('ROC curve')
plt.show()


# STEP 9
# use test set to make predictions
predicted=clf.predict(X_te)
# accuracy score
ac=accuracy_score(predicted,y_te)
# confusion matrix
cm=confusion_matrix(y_te, predicted)
# true positive / false positive rates
# true positives
TP = cm[0][0]
# false positives
FP = cm[0][1]
# false negatives
FN = cm[1][0]
# true negatives
TN = cm[1][1]
# true positive rate
TPR = TP/(TP+FP)
# false positive rate
FPR = FP/(FP+TN)

print('20% TEST DATA:')
print('TRUE POSITVE RATE: ',TPR)
print('FALSE POSITIVE RATE: ',FPR)

