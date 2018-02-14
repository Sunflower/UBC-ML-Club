import csv
import numpy as np
#from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC

train_X = np.load("data/ResNet_train_X.npy")
train_y = np.load("data/ResNet_train_y.npy")
test_X = np.load("data/ResNet_test_X.npy")

# print validation error on best hyperparameters
X_train, X_val, y_train, y_val = train_test_split(train_X, train_y, test_size=0.2)
clf = SVC(C=2.33, tol=1e-6, probability=True)
clf.fit(X_train, y_train)
pred_y = clf.predict(X_val)
print("Accuracy: ", sum(pred_y == y_val)/len(pred_y))
print(classification_report(y_val, pred_y))

# fit with full training set then predict on test set
clf.fit(train_X, train_y)
pred_y = clf.predict(test_X)

with open('predictions.csv', 'w') as preds:
    writer = csv.writer(preds, lineterminator='\n')
    for p in pred_y:
        writer.writerow([p])

