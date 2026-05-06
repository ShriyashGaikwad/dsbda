import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score

df = pd.read_csv("Iris.csv")

df.head()

df.shape

X = df.iloc[:, :-1]
y = df.iloc[:, -1]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=0
)

model = GaussianNB()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

y_pred[:10]

cm = confusion_matrix(y_test, y_pred)
cm

TP = cm[0,0]
FP = cm[:,0].sum() - TP
FN = cm[0,:].sum() - TP
TN = cm.sum() - (TP + FP + FN)

TP, FP, TN, FN

accuracy = accuracy_score(y_test, y_pred)
error_rate = 1 - accuracy
precision = precision_score(y_test, y_pred, average='macro')
recall = recall_score(y_test, y_pred, average='macro')

accuracy, error_rate, precision, recall