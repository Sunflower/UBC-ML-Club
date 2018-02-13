import csv
import numpy as np

from collections import Counter
from nltk import word_tokenize
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import MultinomialNB

# Let's load in the data from our training file 
train_data = [line for line in csv.reader(open('train_data.csv', encoding="utf-8"))]

# Get the sentences
train_sentences = [sentence for sentence,label in train_data]

# Get the labels
labels = np.array([int(label) for sentence,label in train_data])

# Get the test data
test_data = [line for line in csv.reader(open('test_data.csv', encoding="utf-8"))]

# Get the test sentences
test_sentences = [line[0] for line in test_data]

tokenized = []
for i,sent in enumerate(train_sentences):
    tokenized.append(word_tokenize(sent.lower()))

# Let's flatten our list of (list of words) into a list of words
all_words = [word for sent in tokenized for word in sent]

# Identify the 1000 most common words in the corpus and use them as our vocabulary
counter = Counter(all_words)
vocabulary = [word for word,count in counter.most_common(2000)][4:]

# We add an "<UNK>" token to represent all out-of-vocabulary words
vocabulary = ["<UNK>"] + vocabulary

# Now we invert the array to have a mapping of words to indices
word2index = {word:i for i,word in enumerate(vocabulary)}

# Let's create a count vectorization of every sentence. 
# The value at vector[i] will be the number of times vocabulary[i] appears in the sentence.
def count_vectorize(sent):
    vect = np.zeros(len(vocabulary))
    for word in sent:
        vect[word2index.get(word, 0)] += 1
    
    return vect

# Vectorize all of the training data
features = np.stack([count_vectorize(sent) for sent in tokenized])

# Split the data into 95%/5%.
X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.05)

# Hmmm would I really make make the starter file have the best model...
# Check the imports to get an idea of a few other models you could try using.
clf = RandomForestClassifier(n_estimators=25, bootstrap="true")
# clf = MultinomialNB()
clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)

print(classification_report(y_test, y_pred))
print("Accuracy:", sum(y_test == y_pred)/len(y_test))

# Now, let's train on all of the data 
clf.fit(features, labels)

# Prepare the testing data
test_tokenized = [word_tokenize(sent.lower()) for sent in test_sentences]

# Count vectorize the sentences
test_features = np.stack([count_vectorize(sent) for sent in test_tokenized])

y_pred = clf.predict(test_features)

# Write the results to a file
open("predictions.csv", "w+").writelines([str(pred) + "\n" for pred in y_pred])

