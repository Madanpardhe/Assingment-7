import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, classification_report

data_tree = pd.read_csv('suv.csv')


X_tree = data_tree[['Age', 'EstimatedSalary']]
y_tree = data_tree['Purchased']

X_train_tree, X_test_tree, y_train_tree, y_test_tree = train_test_split(X_tree, y_tree, test_size=0.2, random_state=20)

scaler = StandardScaler()
X_train_tree = scaler.fit_transform(X_train_tree)
X_test_tree = scaler.transform(X_test_tree)

dt_entropy = DecisionTreeClassifier(criterion='entropy')
dt_entropy.fit(X_train_tree, y_train_tree)


y_pred_entropy = dt_entropy.predict(X_test_tree)
print("Decision Tree Entropy Confusion Matrix:\n", confusion_matrix(y_test_tree, y_pred_entropy))
print("Decision Tree Entropy Classification Report:\n", classification_report(y_test_tree, y_pred_entropy))

dt_gini = DecisionTreeClassifier(criterion='gini')
dt_gini.fit(X_train_tree, y_train_tree)

y_pred_gini = dt_gini.predict(X_test_tree)
print("Decision Tree Gini Confusion Matrix:\n", confusion_matrix(y_test_tree, y_pred_gini))
print("Decision Tree Gini Classification Report:\n", classification_report(y_test_tree, y_pred_gini))
