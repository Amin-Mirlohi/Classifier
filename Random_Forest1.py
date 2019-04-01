from sklearn.ensemble import RandomForestClassifier
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve
from sklearn.metrics import auc
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_fscore_support
from sklearn.externals import joblib
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

xl = pd.ExcelFile("dataset.xlsx")
df=xl.parse("Sheet1")
features=df.columns[1:42]
label=df['Label_M']
X_train, X_test, Y_train, Y_test = train_test_split(df[features], label, test_size=0.3, random_state=0)
clf = RandomForestClassifier(n_jobs=2, random_state=2017)
clf.fit(X_train[features], Y_train)
joblib.dump(clf, 'RandomForest.pkl')
conf = clf.predict_proba(X_test)
p=clf.predict(X_test)
print(conf)
Prob_being_one=(conf[:,1])
fpr, tpr, thresholds = roc_curve(Y_test, Prob_being_one)
roc_auc = auc(fpr, tpr)
print('roc_auc===>', roc_auc)
print('accu===>', accuracy_score(Y_test, clf.predict(X_test)))
a=precision_recall_fscore_support(Y_test, p, average='binary')
print(a )

print(type(fpr))
sns.set_style("darkgrid")
plt.title('Receiver Operating Characteristic (Random Forest)')
plt.plot(fpr, tpr, 'b', label='AUC = %0.2f'% roc_auc)
plt.legend(loc='lower right')
plt.plot([0,1],[0,1],'r--')
plt.xlim([-0.1,1.2])
plt.ylim([-0.1,1.2])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show()
