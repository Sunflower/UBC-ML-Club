import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC

def dump_labels(labels, filename):
  """
  Dump the labels (given as a numpy array), to a CSV file.
  """
  with open(filename, "w+") as f:
    f.writelines(["%d\n" % label for label in labels])


'''
pandas reads csvs into a dataframe. A dataframe is similar
to a python array or numpy matrix, but is more feature-rich and
handles matrices of mixed types better (e.g. some columns in our data
such as age contain numbers, others such as name contain text).
'''
df = pd.read_csv('train.csv', header=0)

'''
df_X is the dataframe of features we can use for prediction,
which includes all columns excluding "Survived".

Feel free to drop columns of features from this data if
you think omitting them will improve the prediction accuracy.
'''
df_X = df.drop(['Survived'], axis=1)

'''
df_Y is the dataframe of the target we want to predict,
whether or not the passenger survived.
'''
df_Y = df[['Survived']]

'''
You may want to further split the data into training and validation sets.
Search up sklearn's train_test_split for a convenient way to do
this.
'''

'''
Convert the panda dataframes into numpy arrays so it's compatible
as input to sklearn.

For df_Y, we start with 2D matrix (but which only contains 1 column).
The indexing syntax is to extract that 1 column so we end with a flat
1D array.
'''
X = df_X.as_matrix()
y = df_Y.as_matrix()[:,0]

# TODO: train a model on the input data and evaluate the accuracy.
'''
Select a set of features.

Class, Sex, Age, Fare

Convert sex to booleans (1 for male, 0 for female)
'''
features = X[:,[1,3,8]]
features[:,1] = [int(e == 'male') for e in features[:,1]]  

'''
Filter out all rows with missing information.
'''
features = np.nan_to_num(features.astype("float"))

for cell in features[:,1]:
  if cell == 0:
    np.delete(features, cell, 0)

'''
Split X and y into testing and training data (just so we can evaluate
our model before submitting).
'''
X_train, X_test, y_train, y_test = train_test_split(features, y)

'''
Train a random forest classifier with 20 trees.
'''
clf = RandomForestClassifier(n_estimators=20, bootstrap = True,
                            max_depth = 15, max_features=None)
clf.fit(X_train, y_train)

'''
Evaluate our classifier on our training data!
'''
y_pred = clf.predict(X_test) 
print("Accuracy: ", sum(y_pred == y_test)/len(y_pred))
print(classification_report(y_test, y_pred))

'''
Now train on entire training set, and dump test set predictions
to a CSV.

Use the same feature selection/preprocessing code as above (but more 
condensed).
'''
df_test = pd.read_csv('test.csv', header=0)
test_features = df_test.as_matrix()[:,[1,3,8]] 
test_features[:,1] = [int(e == 'male') for e in test_features[:,1]]  
test_features = np.nan_to_num(test_features.astype("float"))

'''
Train the classifier on all the test data, and dump predictions
to a CSV.
'''
clf.fit(features, y)
y_pred = clf.predict(test_features)
dump_labels(y_pred, "predictions.csv")
