import pickle
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import pandas as pd

df = pd.read_csv('diabetes.csv')


## Import ML libraries


# Define our X and y features and split into training/test sets
X = df.drop(['Outcome'], axis=1)
y = df['Outcome']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=109)


X_train.head()


logreg = LogisticRegression(max_iter=1000)
logreg.fit(X_train, y_train)

print(logreg.predict([[1, 103, 30, 38, 83, 43.3, 0.183, 33]]))


# Saving model to disk
# Pickle serializes objects so they can be saved to a file, and loaded in a program again later on.
pickle.dump(logreg, open('model.pkl', 'wb'))
