import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, classification_report

data_svm = pd.read_csv('data_banknote_authentication.csv')

X_svm = data_svm.drop(columns=['class'])
y_svm = data_svm['class']

X_train_svm, X_test_svm, y_train_svm, y_test_svm = train_test_split(X_svm, y_svm, test_size=0.2, random_state=20)

svm_linear = SVC(kernel='linear')
svm_linear.fit(X_train_svm, y_train_svm)

y_pred_linear = svm_linear.predict(X_test_svm)
print("SVM Linear Kernel Confusion Matrix:\n", confusion_matrix(y_test_svm, y_pred_linear))
print("SVM Linear Kernel Classification Report:\n", classification_report(y_test_svm, y_pred_linear))

svm_rbf = SVC(kernel='rbf')
svm_rbf.fit(X_train_svm, y_train_svm)

y_pred_rbf = svm_rbf.predict(X_test_svm)
print("SVM RBF Kernel Confusion Matrix:\n", confusion_matrix(y_test_svm, y_pred_rbf))
print("SVM RBF Kernel Classification Report:\n", classification_report(y_test_svm, y_pred_rbf))

