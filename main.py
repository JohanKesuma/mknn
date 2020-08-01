from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from mknn import MKNN

X, y = load_iris(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
mknn = MKNN(k=3, distance='manhattan')
mknn.fit(X_train, y_train)
predict = mknn.predict(X_test)
print(predict)
    