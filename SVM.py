from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.metrics import roc_curve
from sklearn.metrics import auc
from sklearn import svm
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_fscore_support

xl = pd.ExcelFile("dataset.xlsx")
df=xl.parse("Sheet1")
features=df.columns[2:41]
label=df['Label']
X_train, X_test, Y_train, Y_test = train_test_split(df[features], label, test_size=0.3, random_state=0)
clf=svm.SVC(C=1.0, kernel='poly', degree=3, probability=True, max_iter=-1)
# clf = svm.libsvm()
clf.fit(X_train[features], Y_train)
pre=clf.predict_proba(X_test)
p=clf.predict(X_test)
print(pre)
print(Y_test)
#print(accuracy_score(Y_test, pre))

Prob_being_one=(pre[:,1])
fpr, tpr, thresholds = roc_curve(Y_test, Prob_being_one)
roc_auc = auc(fpr, tpr)
print('roc_auc===>', roc_auc)
print('accu===>', accuracy_score(Y_test, clf.predict(X_test)))
a=precision_recall_fscore_support(Y_test, p, average='binary')
print(a )

plt.title('Receiver Operating Characteristic (Logistic Regression)')
plt.plot(fpr, tpr, 'b', label='AUC = %0.2f'% roc_auc)
plt.legend(loc='lower right')
plt.plot([0,1],[0,1],'r--')
plt.xlim([-0.1,1.2])
plt.ylim([-0.1,1.2])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show()