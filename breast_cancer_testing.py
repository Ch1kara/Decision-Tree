from sklearn import datasets
from sklearn.model_selection import train_test_split
from decision_tree import DecisionTree
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier

data = datasets.load_breast_cancer()
X, y = data.data, data.target

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=2024)

clf = DecisionTree(min_samples_split=5, max_depth=10)
# fit and predict
clf.fit(X_train, y_train)
y_preds = clf.predict(X_test)

acc1 = accuracy_score(y_test, y_preds)
print(f"accuracy score of Decision Tree from Scratch: {acc1}")

# comparison to built in method
clf_entropy = DecisionTreeClassifier(criterion="entropy", random_state=2024, max_depth=10, min_samples_leaf=5)

# fit and predict
clf_entropy.fit(X_train, y_train)
y_preds = clf_entropy.predict(X_test)

acc2 = accuracy_score(y_preds, y_test)

print(f"accuracy score of built-in Sklearn Decision Tree: {acc2}")





